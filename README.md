<div align="center">

## 🎙️ AI Podcast Generator

*Powered by VibeVoice, Gemini, and Whisper*

[![Original VibeVoice](https://img.shields.io/badge/Based%20on-VibeVoice-blue?logo=microsoft)](https://github.com/microsoft/VibeVoice)
[![VibeVoice Paper](https://img.shields.io/badge/Technical-Report-red?logo=adobeacrobatreader)](https://arxiv.org/pdf/2508.19205)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)

</div>

Generate AI-powered podcasts from any topic with multi-speaker conversations, natural speech synthesis, and automatic video generation.

## 🎬 Demo

<div align="center">


*Example podcast generated about AI and natural world technologies*

https://github.com/user-attachments/assets/09bcc17e-9435-4989-9ffa-4ca6e43983df



</div>

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/podcast-generator.git
cd podcast-generator
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
python main.py
```

The generator will create:

- `podcast_script.txt` - Generated conversation script
- `podcast_output.wav` - Final audio file  
- `podcast_subtitles.srt` - Subtitle file
- `podcast_video.mp4` - Complete video with subtitles

## 📁 Project Structure

```text
├── src/
│   ├── podcast_generator.py    # Main orchestrator
│   ├── script_generator.py     # Gemini-powered script generation
│   ├── audio_processor.py      # VibeVoice TTS processing  
│   ├── subtitle_generator.py   # Whisper-powered subtitles
│   ├── video_generator.py      # Video composition
│   └── character_manager.py    # Speaker personality management
├── characters.json             # Speaker voice configurations
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## 🔧 Requirements

- NVIDIA GPU with 8GB+ VRAM
- Python 3.8+
- Gemini API key
- ~10GB storage

**Environment Variables:**
```bash
GEMINI_API_KEY=your_api_key
PODCAST_TOPIC="Your topic"
PODCAST_DURATION_MINUTES=5
```

## 📄 License

MIT License - Built with [VibeVoice](https://github.com/microsoft/VibeVoice), [Gemini](https://ai.google.dev/gemini-api), and [Whisper](https://github.com/openai/whisper).

**Use responsibly:** Disclose AI-generated content and verify accuracy.
