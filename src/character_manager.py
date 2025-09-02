
import json
import librosa
from pathlib import Path
from typing import List, Tuple, Dict

from .config import VOICES_DIR, VOICE_SAMPLE_RATE
from .audio_processor import optimize_voice_snippet


class CharacterManager:
    
    def __init__(self):
        self.speaker_voices = {}
    
    def load_characters(self, config_path: str):
        with open(config_path, "r") as f:
            characters = json.load(f)

        self.speaker_voices = {}
        for char in characters:
            voice_file_path = Path(VOICES_DIR) / f"{char['slug']}.wav"
            
            optimized_voice = None
            if voice_file_path.exists():
                try:
                    voice_data, _ = librosa.load(
                        str(voice_file_path), sr=VOICE_SAMPLE_RATE, mono=True
                    )
                    optimized_voice = optimize_voice_snippet(voice_data)
                except Exception:
                    pass
            
            self.speaker_voices[char["name"]] = {
                "slug": char["slug"],
                "voice_file": f"{char['slug']}.wav",
                "has_voice_sample": voice_file_path.exists(),
                "persona": char["persona"],
                "optimized_voice": optimized_voice,
            }
    
    def get_available_characters(self) -> List[str]:
        return [
            name for name, info in self.speaker_voices.items() 
            if info["has_voice_sample"]
        ]
    
    def map_speakers_to_characters(self, parsed_script: List[Tuple[int, str]]) -> Dict[int, str]:
        script_speakers = list(set(speaker_id for speaker_id, _ in parsed_script))
        available_chars = self.get_available_characters()
        
        if len(script_speakers) > len(available_chars):
            print(f"Warning: Script uses {len(script_speakers)} speakers but only {len(available_chars)} voice samples available")
        
        speaker_mapping = {}
        for i, speaker_id in enumerate(sorted(script_speakers)):
            if i < len(available_chars):
                char_name = available_chars[i]
                speaker_mapping[speaker_id] = char_name
                print(f"Speaker {speaker_id + 1} → {char_name}")
            else:
                print(f"Warning: No voice sample for Speaker {speaker_id + 1}")
        
        return speaker_mapping
    
    def get_voice_paths(self, speaker_mapping: Dict[int, str]) -> List[str]:
        voice_paths = []
        for speaker_id in sorted(speaker_mapping.keys()):
            char_name = speaker_mapping[speaker_id]
            voice_file = self.speaker_voices[char_name]["voice_file"]
            voice_paths.append(str(Path(VOICES_DIR) / voice_file))
        
        return voice_paths