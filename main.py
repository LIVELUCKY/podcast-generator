"""
Podcast Generator - AI-powered podcast creation with voice synthesis and subtitles.
"""

import os
import warnings
import asyncio

# Setup environment before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging
from dotenv import load_dotenv

# Configure logging and environment
hf_logging.set_verbosity_error()
load_dotenv()

from src.podcast_generator import PodcastGenerator
from src.config import PODCAST_TOPIC, PODCAST_DURATION_MINUTES


async def main():
    """Main entry point for podcast generation."""
    generator = PodcastGenerator()
    await generator.generate_podcast()


if __name__ == "__main__":
    if not all([PODCAST_TOPIC, PODCAST_DURATION_MINUTES]):
        print("Error: PODCAST_TOPIC and PODCAST_DURATION_MINUTES must be set in the .env file.")
    else:
        asyncio.run(main())