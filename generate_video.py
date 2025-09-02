#!/usr/bin/env python3
"""Generate video with dynamic speaker switching."""

import sys
from pathlib import Path
from src.video_generator import VideoGenerator
from src.config import (
    FINAL_OUTPUT_FILENAME,
    SRT_FILENAME,
    CONFIG_FILE,
    VIDEO_OUTPUT_FILE
)


def main():
    """Main video generation function."""
    enable_subtitles = "--no-subtitles" not in sys.argv
    
    # Check required files
    audio_path = Path(FINAL_OUTPUT_FILENAME)
    srt_path = Path(SRT_FILENAME)
    config_path = Path(CONFIG_FILE)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if enable_subtitles and not srt_path.exists():
        print(f"Error: SRT file not found: {srt_path}")
        sys.exit(1)
    
    # Generate video
    generator = VideoGenerator()
    
    print("🎬 Generating video with dynamic speaker switching...")
    success = generator.generate_video(
        audio_path=str(audio_path),
        srt_path=str(srt_path),
        config_path=str(config_path),
        output_path=VIDEO_OUTPUT_FILE,
        enable_subtitles=enable_subtitles
    )
    
    if success:
        print("✅ Video generation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Video generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()