"""Main podcast generator orchestrating all components."""

import time
import asyncio
import gc
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from .device_manager import DeviceManager
from .audio_processor import AudioProcessor
from .script_generator import ScriptGenerator
from .subtitle_generator import SubtitleGenerator
from .character_manager import CharacterManager
from .config import (
    VIBEVOICE_MODEL_PATH, CHARACTERS_CONFIG_FILE, RAW_OUTPUT_FILENAME,
    FINAL_OUTPUT_FILENAME, SRT_FILENAME, PODCAST_TOPIC, PODCAST_DURATION_MINUTES
)


def download_model(model_path: str):
    """Download VibeVoice model if it doesn't exist."""
    if not (Path(model_path) / "config.json").exists():
        snapshot_download(
            repo_id="microsoft/VibeVoice-1.5B",
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,
        )


class PodcastGenerator:
    """Main podcast generator coordinating all components."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.character_manager = CharacterManager()
        self.audio_processor = AudioProcessor(self.device_manager)
        self.subtitle_generator = SubtitleGenerator(self.device_manager)
        self.script_generator = None  # Initialize after characters are loaded
    
    async def generate_podcast(self):
        """Generate complete podcast with audio and subtitles."""
        start_time = time.time()

        try:
            # Initialize components
            await self._initialize()
            
            # Generate script
            script_text = await self.script_generator.generate_script(
                PODCAST_TOPIC, PODCAST_DURATION_MINUTES
            )
            if not script_text.strip():
                raise ValueError("Script is empty. Cannot proceed.")

            # Load models and synthesize audio
            self.audio_processor.load_vibevoice_model(VIBEVOICE_MODEL_PATH)
            parsed_script = self._parse_script(script_text)
            speaker_mapping = self.character_manager.map_speakers_to_characters(parsed_script)
            voice_paths = self.character_manager.get_voice_paths(speaker_mapping)

            duration = await asyncio.to_thread(
                self.audio_processor.synthesize_audio, 
                script_text, voice_paths, RAW_OUTPUT_FILENAME
            )

            self.audio_processor.cleanup()

            # Post-process and create subtitles
            if duration > 1.0:
                await self.audio_processor.post_process_audio(
                    RAW_OUTPUT_FILENAME, FINAL_OUTPUT_FILENAME
                )
                await self.subtitle_generator.generate_subtitles(
                    FINAL_OUTPUT_FILENAME, parsed_script, speaker_mapping
                )

            self._print_results(start_time, duration)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self._cleanup_all()

    async def _initialize(self):
        """Initialize all components."""
        await asyncio.gather(
            asyncio.to_thread(download_model, VIBEVOICE_MODEL_PATH),
            asyncio.to_thread(self.character_manager.load_characters, CHARACTERS_CONFIG_FILE),
        )
        
        # Initialize script generator after characters are loaded
        self.script_generator = ScriptGenerator(self.character_manager.speaker_voices)

    def _parse_script(self, script_text: str):
        """Parse script using VibeVoice processor."""
        # We need to load the processor temporarily just to parse
        self.audio_processor.load_vibevoice_model(VIBEVOICE_MODEL_PATH)
        _, processor = self.audio_processor.vibevoice_model, self.audio_processor.vibevoice_processor
        return processor._parse_script(script_text)

    def _print_results(self, start_time: float, duration: float):
        """Print generation results."""
        total_time = time.time() - start_time
        rtf = total_time / duration if duration > 0 else 0
        print(f"✅ Podcast created: {FINAL_OUTPUT_FILENAME} ({duration:.1f}s)")
        print(f"✅ Subtitles: {SRT_FILENAME}")
        print(f"⚡ Total time: {total_time:.2f}s (RTF: {rtf:.2f}x)")

    def _cleanup_all(self):
        """Clean up all components."""
        self.audio_processor.cleanup()
        self.subtitle_generator.cleanup()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()