"""Subtitle generation using script text with intelligent timing."""

import time
import asyncio
import gc
import torch
import librosa
import whisper
import textwrap
import re
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from thefuzz import fuzz, process

from .config import WHISPER_MODEL, WHISPER_LANGUAGE, SRT_FILENAME, FUZZY_MATCH_THRESHOLD


@dataclass
class SubtitleSegment:
    start_time: float
    end_time: float
    text: str
    speaker: str = None


class SubtitleGenerator:
    """Generates subtitles using script text with Whisper timing."""

    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.whisper_model = None

    def load_whisper_model(self):
        """Load Whisper model for timing extraction."""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(
                WHISPER_MODEL,
                device=self.device_manager.whisper_device,
                download_root="./models/whisper"
            )
        return self.whisper_model

    async def generate_subtitles(
        self,
        audio_path: str,
        parsed_script: List[Tuple[int, str]],
        speaker_mapping: dict
    ) -> List[SubtitleSegment]:
        """Generate subtitles using script text with Whisper timing and fuzzy matching."""
        model = self.load_whisper_model()

        try:
            audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)

            # Use Whisper with word-level timestamps for precise alignment
            result = await asyncio.to_thread(
                model.transcribe,
                audio_data,
                language=WHISPER_LANGUAGE,
                fp16=(self.device_manager.whisper_device == "cuda"),
                verbose=False,
                word_timestamps=True,  # Enable word-level timing
            )

            self.cleanup()

            # Get detailed timing from Whisper
            whisper_segments = [
                SubtitleSegment(start_time=s["start"], end_time=s["end"], text=s["text"].strip())
                for s in result["segments"] if s["text"].strip()
            ]

            # Create aligned subtitles using fuzzy matching
            script_subtitles = self._create_aligned_subtitles(
                result, parsed_script, speaker_mapping
            )

            # Validate and refine alignment
            self._validate_and_refine_alignment(whisper_segments, script_subtitles)

            self._write_srt(script_subtitles, Path(SRT_FILENAME))
            self._write_speaker_mapping(script_subtitles)
            return script_subtitles
        except Exception as e:
            print(f"Transcription failed: {e}")
            return []

    def _create_aligned_subtitles(
        self,
        whisper_result: dict,
        script_segments: List[Tuple[int, str]],
        speaker_mapping: dict
    ) -> List[SubtitleSegment]:
        """Create aligned subtitle segments using fuzzy matching and shorter sentences."""
        if not script_segments:
            return []

        # Break script into shorter, readable chunks
        script_chunks = self._break_into_readable_chunks(script_segments, speaker_mapping)

        if not script_chunks:
            return []

        # Get word-level timing from Whisper
        word_timings = []
        for segment in whisper_result.get("segments", []):
            for word_info in segment.get("words", []):
                if word_info.get("word", "").strip():
                    word_timings.append({
                        "word": word_info["word"].strip().lower(),
                        "start": word_info["start"],
                        "end": word_info["end"]
                    })

        # Align script chunks with Whisper word timings using fuzzy matching
        aligned_subtitles = self._align_with_sequence_matching(script_chunks, word_timings)

        return aligned_subtitles

    def _break_into_readable_chunks(
        self,
        script_segments: List[Tuple[int, str]],
        speaker_mapping: dict
    ) -> List[dict]:
        """Break script into shorter, more readable chunks."""
        chunks = []

        for speaker_id, text in script_segments:
            speaker_name = speaker_mapping.get(speaker_id, f"Speaker {speaker_id + 1}")
            clean_text = text.strip()

            if not clean_text:
                continue

            # Split by natural sentence boundaries
            sentences = re.split(r'[.!?]+', clean_text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Further split long sentences by commas or natural pauses
                if len(sentence) > 120:  # If sentence is too long
                    parts = re.split(r'[,;:]+', sentence)
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 10:  # Minimum meaningful length
                            chunks.append({
                                "speaker": speaker_name,
                                "text": part,
                                "speaker_id": speaker_id,
                                "word_count": len(part.split())
                            })
                else:
                    chunks.append({
                        "speaker": speaker_name,
                        "text": sentence,
                        "speaker_id": speaker_id,
                        "word_count": len(sentence.split())
                    })

        return chunks

    def _align_with_sequence_matching(
        self,
        script_chunks: List[dict],
        word_timings: List[dict]
    ) -> List[SubtitleSegment]:
        """Align script chunks with Whisper word timings using a robust sequence matching search."""
        if not word_timings:
            return self._fallback_timing(script_chunks)

        aligned_subtitles = []
        whisper_words = [w["word"] for w in word_timings]
        word_cursor = 0

        for chunk in script_chunks:
            clean_chunk_text = re.sub(r'[^\w\s]', '', chunk["text"]).lower().strip()
            if not clean_chunk_text:
                continue

            chunk_words = clean_chunk_text.split()
            chunk_word_count = len(chunk_words)

            # Define a search window in the whisper transcript
            search_start = max(0, word_cursor - 10)
            search_end = min(len(whisper_words), word_cursor + chunk_word_count + 20)
            search_window_words = whisper_words[search_start:search_end]

            # Use SequenceMatcher to find the best match
            matcher = SequenceMatcher(None, search_window_words, chunk_words, autojunk=False)
            match = matcher.find_longest_match(0, len(search_window_words), 0, chunk_word_count)

            # A good match should be reasonably long
            is_good_match = match.size > 0 and (match.size / chunk_word_count) > 0.4

            if is_good_match:
                start_word_index = search_start + match.a
                end_word_index = min(start_word_index + chunk_word_count, len(word_timings) - 1)
                
                start_time = word_timings[start_word_index]["start"]
                end_time = word_timings[end_word_index]["end"]
                
                word_cursor = end_word_index + 1
            else:
                # Fallback: estimate timing if no good match is found
                last_end = aligned_subtitles[-1].end_time if aligned_subtitles else (word_timings[word_cursor - 1]["end"] if word_cursor > 0 else 0)
                start_time = last_end
                duration = max(1.5, chunk_word_count / 2.5)  # Estimate 2.5 words/sec
                end_time = start_time + duration
                word_cursor = min(word_cursor + chunk_word_count, len(word_timings) - 1)

            aligned_subtitles.append(SubtitleSegment(
                start_time=start_time,
                end_time=end_time,
                text=chunk["text"],
                speaker=chunk["speaker"]
            ))

        # Post-process to fix overlaps and durations
        for i, sub in enumerate(aligned_subtitles):
            if i > 0:
                sub.start_time = max(sub.start_time, aligned_subtitles[i - 1].end_time + 0.01)
            
            duration = sub.end_time - sub.start_time
            if duration < 1.0:
                sub.end_time = sub.start_time + 1.0
            elif duration > 10.0:
                sub.end_time = sub.start_time + 10.0

        return aligned_subtitles

    def _fallback_timing(self, script_chunks: List[dict]) -> List[SubtitleSegment]:
        """Fallback timing when word-level alignment fails."""
        if not script_chunks:
            return []

        total_words = sum(chunk["word_count"] for chunk in script_chunks)
        estimated_duration = total_words / 2.5  # Assume 2.5 words per second

        subtitles = []
        current_time = 0.0

        for chunk in script_chunks:
            duration = max(1.5, min(8.0, chunk["word_count"] / 2.5))

            subtitles.append(SubtitleSegment(
                start_time=current_time,
                end_time=current_time + duration,
                text=chunk["text"],
                speaker=chunk["speaker"]
            ))

            current_time += duration

        return subtitles

    def _validate_and_refine_alignment(
        self,
        whisper_segments: List[SubtitleSegment],
        script_segments: List[SubtitleSegment]
    ):
        """Validate and refine alignment using advanced fuzzy matching."""
        if not whisper_segments or not script_segments:
            return

        # Combine all text for comparison
        whisper_text = " ".join(seg.text.lower().strip() for seg in whisper_segments)
        script_text = " ".join(seg.text.lower().strip() for seg in script_segments)

        # Clean up text for comparison
        whisper_clean = re.sub(r'[^\w\s]', '', whisper_text)
        script_clean = re.sub(r'[^\w\s]', '', script_text)

        # Use multiple fuzzy matching algorithms
        similarity_ratio = fuzz.ratio(whisper_clean, script_clean)
        partial_ratio = fuzz.partial_ratio(whisper_clean, script_clean)
        token_sort_ratio = fuzz.token_sort_ratio(whisper_clean, script_clean)

        # Take the best score
        best_similarity = max(similarity_ratio, partial_ratio, token_sort_ratio) / 100.0

        print(f"✓ Subtitle alignment quality: {best_similarity:.1%}")
        print(f"  - Ratio: {similarity_ratio}%")
        print(f"  - Partial: {partial_ratio}%")
        print(f"  - Token Sort: {token_sort_ratio}%")

        if best_similarity < FUZZY_MATCH_THRESHOLD:
            print(f"⚠️  Warning: Script alignment below threshold")
            print(f"   Best similarity: {best_similarity:.2%} (threshold: {FUZZY_MATCH_THRESHOLD:.0%})")

            # Show sample differences for debugging
            if len(whisper_clean) > 100 and len(script_clean) > 100:
                print(f"   Whisper sample: '{whisper_clean[:100]}...'")
                print(f"   Script sample:  '{script_clean[:100]}...'")
        else:
            print(f"✓ Subtitle alignment successful")

    def _write_srt(self, segments: List[SubtitleSegment], output_path: Path):
        """Write subtitle segments to SRT file with improved formatting."""
        def format_time(seconds: float) -> str:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                text = f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text

                # Break long lines for better readability (max 50 chars per line)
                lines = textwrap.wrap(text, width=50)

                # Format speaker name in bold
                if lines and seg.speaker and ":" in lines[0]:
                    name, rest = lines[0].split(":", 1)
                    lines[0] = f"<b>{name}:</b>{rest}"

                f.write(f"{i}\n")
                f.write(f"{format_time(seg.start_time)} --> {format_time(seg.end_time)}\n")
                f.write("\n".join(lines) + "\n\n")

    def _write_speaker_mapping(self, segments: List[SubtitleSegment]):
        """Write speaker timing mapping for video generation with full audio coverage."""
        import json
        
        if not segments:
            return
        
        speaker_timeline = []
        first_speaker = segments[0].speaker
        last_speaker = None
        last_end_time = 0.0
        audio_start = 0.0
        
        # Start from beginning with first speaker (even before they speak)
        current_time = audio_start
        
        for seg in segments:
            # If speaker changed, create a timing segment
            if seg.speaker != last_speaker:
                if last_speaker is not None:
                    # End previous speaker's segment at start of new speaker
                    speaker_timeline.append({
                        'start': round(last_start_time, 2),
                        'end': round(seg.start_time, 2),
                        'speaker': last_speaker
                    })
                else:
                    # Very first segment - show first speaker from beginning
                    if seg.start_time > audio_start:
                        speaker_timeline.append({
                            'start': round(audio_start, 2),
                            'end': round(seg.start_time, 2),
                            'speaker': first_speaker
                        })
                
                # Start new speaker's segment  
                last_speaker = seg.speaker
                last_start_time = seg.start_time
            
            last_end_time = max(last_end_time, seg.end_time)
        
        # Add final segment - show last speaker until end of audio
        if last_speaker is not None:
            speaker_timeline.append({
                'start': round(last_start_time, 2),
                'end': round(last_end_time, 2),
                'speaker': last_speaker
            })
        
        # Write to JSON file
        with open('speaker_timeline.json', 'w') as f:
            json.dump(speaker_timeline, f, indent=2)
        
        print(f"✓ Created speaker timeline with {len(speaker_timeline)} segments covering full audio")

    def cleanup(self):
        """Clean up Whisper model."""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()