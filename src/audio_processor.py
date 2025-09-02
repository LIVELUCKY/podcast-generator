import asyncio
import shutil
import psutil
import re
import torch
import librosa
import soundfile as sf
import whisper
import numpy as np
from pathlib import Path
from typing import List, Tuple
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from .config import (
    VOICE_SAMPLE_RATE,
    DDMP_INFERENCE_STEPS,
    CFG_SCALE,
    SAMPLING_TEMPERATURE,
    MAX_NEW_TOKENS,
    VOICE_SNIPPET_MIN_DURATION,
    VOICE_SNIPPET_MAX_DURATION,
    AUDIO_QUALITY_THRESHOLD,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
)


def optimize_voice_snippet(audio_data, target_duration=None):
    if target_duration is None:
        target_duration = min(
            max(len(audio_data) / VOICE_SAMPLE_RATE, VOICE_SNIPPET_MIN_DURATION),
            VOICE_SNIPPET_MAX_DURATION,
        )

    target_samples = int(target_duration * VOICE_SAMPLE_RATE)
    current_samples = len(audio_data)

    if current_samples < target_samples:
        return audio_data

    if current_samples > target_samples * 2:
        energy = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
        energy_smooth = librosa.util.normalize(energy)

        high_energy_frames = energy_smooth > AUDIO_QUALITY_THRESHOLD
        if high_energy_frames.sum() > 0:
            frame_indices = librosa.frames_to_samples(range(len(high_energy_frames)))
            valid_regions = frame_indices[high_energy_frames]
            if len(valid_regions) > 0:
                start_sample = valid_regions[0]
                end_sample = min(start_sample + target_samples, current_samples)
                return audio_data[start_sample:end_sample]

    mid_point = current_samples // 2
    half_target = target_samples // 2
    start = max(0, mid_point - half_target)
    end = min(current_samples, start + target_samples)

    return audio_data[start:end]


def detect_problematic_audio(
    audio_data: np.ndarray, sample_rate: int, whisper_model
) -> Tuple[float, float]:
    duration = len(audio_data) / sample_rate

    #  first 3 seconds
    start_seconds = min(3.0, duration / 3)
    start_samples = int(start_seconds * sample_rate)
    start_audio = audio_data[:start_samples]

    #  last 3 seconds
    end_seconds = min(3.0, duration / 3)
    end_samples = int(end_seconds * sample_rate)
    end_audio = audio_data[-end_samples:]

    problematic_words = [
        "uh",
        "um",
        "ah",
        "er",
        "hmm",
        "mmm",
        "uhh",
        "umm",
        "ahh",
        "[",
        "]",
        "(",
        ")",
        "<",
        ">",
    ]

    start_cut = 0.0
    end_cut = 0.0

    try:
        if len(start_audio) > sample_rate * 0.5:  # At least 0.5 seconds
            start_result = whisper_model.transcribe(
                start_audio, language=WHISPER_LANGUAGE, fp16=False, verbose=False
            )
            start_text = start_result["text"].lower().strip()
            # Only cut if text is ONLY problematic words or completely empty
            words = start_text.split()
            if (
                len(start_text.strip()) == 0 or 
                (len(words) > 0 and all(word in problematic_words for word in words))
            ):
                start_cut = start_seconds  # Cut the entire problematic section

        if len(end_audio) > sample_rate * 0.5:
            end_result = whisper_model.transcribe(
                end_audio, language=WHISPER_LANGUAGE, fp16=False, verbose=False
            )
            end_text = end_result["text"].lower().strip()

            # Only cut if text is ONLY problematic words or completely empty
            end_words = end_text.split()
            if (
                len(end_text.strip()) == 0 or 
                (len(end_words) > 0 and all(word in end_words for word in problematic_words))
            ):
                end_cut = end_seconds  # Cut the entire problematic section

    except Exception:
        # If whisper analysis fails, apply conservative cuts
        start_cut = min(0.5, start_seconds * 0.3)
        end_cut = min(0.5, end_seconds * 0.3)

    return start_cut, end_cut


def add_silence_padding(
    audio_data: np.ndarray,
    sample_rate: int,
    start_silence: float = 0.5,
    end_silence: float = 0.5,
) -> np.ndarray:
    start_samples = int(start_silence * sample_rate)
    end_samples = int(end_silence * sample_rate)

    start_padding = np.zeros(start_samples, dtype=audio_data.dtype)
    end_padding = np.zeros(end_samples, dtype=audio_data.dtype)

    return np.concatenate([start_padding, audio_data, end_padding])


class AudioProcessor:

    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.vibevoice_model = None
        self.vibevoice_processor = None
        self.whisper_model = None

    def load_vibevoice_model(self, model_path: str):
        if self.vibevoice_model is not None:
            return self.vibevoice_model, self.vibevoice_processor

        device = self.device_manager.device
        dtype = self.device_manager.dtype
        config = self.device_manager.config

        load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

        if device == "cuda":
            load_kwargs.update(
                {
                    "attn_implementation": (
                        "flash_attention_2"
                        if dtype in [torch.bfloat16, torch.float16]
                        else "sdpa"
                    ),
                    "device_map": "auto",
                }
            )
        elif device == "mps":
            load_kwargs.update({"attn_implementation": "sdpa", "device_map": device})

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, **load_kwargs
        )
        model.eval()

        if device == "cuda" and config.use_channels_last:
            model = model.to(memory_format=torch.channels_last)

        inference_steps = DDMP_INFERENCE_STEPS if device != "cpu" else 15
        model.set_ddpm_inference_steps(inference_steps)

        processor = VibeVoiceProcessor.from_pretrained(
            model_path, db_normalize=True, speech_tok_compress_ratio=2400
        )

        self.vibevoice_model = model
        self.vibevoice_processor = processor

        return model, processor

    def synthesize_audio(
        self, script_text: str, voice_samples: List[str], output_path: str
    ) -> float:
        model, processor = self.vibevoice_model, self.vibevoice_processor

        inputs = processor(
            text=[script_text], voice_samples=voice_samples, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device_manager.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generation_args = {
            "tokenizer": processor.tokenizer,
            "cfg_scale": CFG_SCALE,
            "do_sample": True,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": 0.9,
            "max_new_tokens": MAX_NEW_TOKENS,
            "verbose": True,
            "show_progress_bar": True,
        }

        with torch.no_grad():
            if (
                self.device_manager.config.use_amp
                and self.device_manager.device == "cuda"
            ):
                with torch.cuda.amp.autocast("cuda", dtype=self.device_manager.dtype):
                    outputs = model.generate(**inputs, **generation_args)
            else:
                outputs = model.generate(**inputs, **generation_args)

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio_tensor = outputs.speech_outputs[0].float().cpu().numpy().flatten()

            audio_tensor = self._apply_audio_cleanup(audio_tensor)

            sf.write(output_path, audio_tensor, VOICE_SAMPLE_RATE, subtype="PCM_16")
            return len(audio_tensor) / VOICE_SAMPLE_RATE

        return 0.0

    async def post_process_audio(self, input_path: str, output_path: str):
        if not shutil.which("ffmpeg") or not Path(input_path).exists():
            if Path(input_path).exists():
                shutil.copy(input_path, output_path)
            return

        audio_filters = "afftdn=nr=10:nf=-25,dynaudnorm=f=150:g=15"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            audio_filters,
            "-ar",
            str(VOICE_SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-threads",
            str(psutil.cpu_count()),
            output_path,
            "-loglevel",
            "error",
        ]

        try:
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed with code {process.returncode}")
        except Exception:
            shutil.copy(input_path, output_path)

    def _apply_audio_cleanup(self, audio_data: np.ndarray) -> np.ndarray:
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(
                WHISPER_MODEL,
                device="cpu",  # Use CPU to avoid memory conflicts
                download_root="./models/whisper",
            )

        try:
            start_cut, end_cut = detect_problematic_audio(
                audio_data, VOICE_SAMPLE_RATE, self.whisper_model
            )

            if start_cut > 0 or end_cut > 0:
                start_samples = int(start_cut * VOICE_SAMPLE_RATE)
                end_samples = int(end_cut * VOICE_SAMPLE_RATE)

                start_idx = start_samples
                end_idx = (
                    len(audio_data) - end_samples if end_cut > 0 else len(audio_data)
                )

                if start_idx < end_idx:
                    audio_data = audio_data[start_idx:end_idx]

            audio_data = add_silence_padding(audio_data, VOICE_SAMPLE_RATE, start_silence=2.0, end_silence=2.0)

            return audio_data

        except Exception:
            return add_silence_padding(audio_data, VOICE_SAMPLE_RATE, start_silence=2.0, end_silence=2.0)

    def cleanup(self):
        self.vibevoice_model = None
        self.vibevoice_processor = None
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
