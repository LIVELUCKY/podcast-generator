"""Configuration constants for the podcast generator."""

import os

# Audio parameters
VOICE_SAMPLE_RATE = 24000
DDMP_INFERENCE_STEPS = 20
CFG_SCALE = 1.7
SAMPLING_TEMPERATURE = 0.8
MAX_NEW_TOKENS = None

# Voice snippet optimization
VOICE_SNIPPET_MIN_DURATION = 3.0
VOICE_SNIPPET_MAX_DURATION = 15.0
AUDIO_QUALITY_THRESHOLD = 0.1

# Model paths
VIBEVOICE_MODEL_PATH = os.getenv("VIBEVOICE_MODEL_PATH", "./models/VibeVoice-1.5B")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Script validation
FUZZY_MATCH_THRESHOLD = 0.9  # Minimum similarity ratio for script validation
SCRIPT_LENGTH_TOLERANCE = float(os.getenv("SCRIPT_LENGTH_TOLERANCE", "0.9"))  # 90% of target is acceptable
SCRIPT_MAX_LENGTH_MULTIPLIER = float(os.getenv("SCRIPT_MAX_LENGTH_MULTIPLIER", "1.5"))  # Max 150% of target
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "600"))  # Words per chunk for LLM generation

# Content parameters
PODCAST_TOPIC = os.getenv("PODCAST_TOPIC")
PODCAST_DURATION_MINUTES = int(os.getenv("PODCAST_DURATION_MINUTES", 1))
WORDS_PER_MINUTE = 130

# File names
CHARACTERS_CONFIG_FILE = "characters.json"
VOICES_DIR = "media"
RAW_OUTPUT_FILENAME = "podcast_raw_output.wav"
FINAL_OUTPUT_FILENAME = "podcast_output.wav"
SRT_FILENAME = "podcast_subtitles.srt"
SCRIPT_CACHE_FILE = "podcast_script.txt"

# Video Generation Settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_OUTPUT_FILE = "podcast_video.mp4"
IMAGES_DIR = "images"
CONFIG_FILE = "characters.json"