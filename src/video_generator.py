"""Video generation with dynamic speaker switching."""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from .config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_OUTPUT_FILE,
    SRT_FILENAME,
    CONFIG_FILE,
    IMAGES_DIR,
)


class VideoGenerator:
    """Handles video generation with speaker switching based on SRT timing."""
    
    def __init__(self):
        self.width = VIDEO_WIDTH
        self.height = VIDEO_HEIGHT
        self.temp_dir = None
    
    def generate_video(
        self,
        audio_path: str,
        srt_path: str,
        config_path: str,
        output_path: str,
        enable_subtitles: bool = True
    ) -> bool:
        """Generate video with dynamic speaker switching."""
        try:
            # Load speaker configuration
            with open(config_path, 'r') as f:
                speakers = json.load(f)
            
            if not speakers:
                print("Error: No speakers found in config")
                return False
            
            # Get audio duration
            audio_duration = self._get_audio_duration(audio_path)
            if audio_duration <= 0:
                print("Error: Invalid audio duration")
                return False
            
            # Load speaker timeline mapping
            speaker_timeline = self._load_speaker_timeline()
            if not speaker_timeline:
                print("Error: No speaker timeline found")
                return False
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                
                # Pre-process speaker images into standardized video clips
                video_clips = self._create_speaker_clips(speakers, audio_duration)
                
                # Generate the final video
                return self._assemble_final_video(
                    video_clips, 
                    speaker_timeline, 
                    audio_path, 
                    output_path,
                    enable_subtitles,
                    srt_path
                )
                
        except Exception as e:
            print(f"Video generation failed: {e}")
            return False
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', 
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ], capture_output=True, text=True, check=True)
            
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Failed to get audio duration: {e}")
            return 0.0
    
    def _parse_srt(self, srt_path: str, speakers: List[Dict]) -> List[Dict]:
        """Parse SRT file and extract consolidated speaker timing information."""
        raw_timeline = []
        
        # Create speaker name to slug mapping
        speaker_map = {speaker['name']: speaker['slug'] for speaker in speakers}
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT using regex
            srt_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n<b>([^<]+):</b>'
            
            for match in re.finditer(srt_pattern, content, re.MULTILINE):
                segment_num, start_str, end_str, speaker_name = match.groups()
                
                start_seconds = self._time_to_seconds(start_str)
                end_seconds = self._time_to_seconds(end_str)
                
                # Find speaker slug
                speaker_slug = speaker_map.get(speaker_name, speakers[0]['slug'])
                
                raw_timeline.append({
                    'start': start_seconds,
                    'end': end_seconds,
                    'speaker': speaker_name,
                    'slug': speaker_slug
                })
            
            # Sort by start time
            raw_timeline.sort(key=lambda x: x['start'])
            
            # Consolidate consecutive segments from the same speaker
            consolidated_timeline = self._consolidate_speaker_segments(raw_timeline)
            
            return consolidated_timeline
            
        except Exception as e:
            print(f"Failed to parse SRT: {e}")
            return []
    
    def _consolidate_speaker_segments(self, raw_timeline: List[Dict]) -> List[Dict]:
        """Consolidate consecutive segments from the same speaker into longer speaking turns."""
        if not raw_timeline:
            return []
        
        consolidated = []
        current_segment = None
        
        # Group consecutive segments by the same speaker
        for segment in raw_timeline:
            if current_segment is None:
                # First segment
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': segment['speaker'],
                    'slug': segment['slug']
                }
            elif (current_segment['slug'] == segment['slug'] and 
                  abs(segment['start'] - current_segment['end']) <= 2.0):  # Allow up to 2 second gap
                # Same speaker and close timing - extend current segment
                current_segment['end'] = segment['end']
            else:
                # Different speaker or large gap - save current and start new
                consolidated.append(current_segment)
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': segment['speaker'],
                    'slug': segment['slug']
                }
        
        # Don't forget the last segment
        if current_segment:
            consolidated.append(current_segment)
        
        # Log the consolidation results
        print(f"📊 Consolidated {len(raw_timeline)} subtitle fragments into {len(consolidated)} speaking turns:")
        for i, seg in enumerate(consolidated):
            duration = seg['end'] - seg['start']
            print(f"  {i+1}. {seg['speaker']}: {seg['start']:.2f}-{seg['end']:.2f}s ({duration:.2f}s)")
        
        return consolidated
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds."""
        # Format: HH:MM:SS,mmm
        time_parts = time_str.replace(',', '.').split(':')
        hours = float(time_parts[0])
        minutes = float(time_parts[1])
        seconds = float(time_parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _load_speaker_timeline(self) -> List[Dict]:
        """Load speaker timeline mapping from JSON file."""
        timeline_file = Path("speaker_timeline.json")
        if not timeline_file.exists():
            print("Error: speaker_timeline.json not found")
            return []
        
        try:
            with open(timeline_file, 'r') as f:
                timeline = json.load(f)
            
            # Convert speaker names to slugs for video clips
            speaker_map = {}
            if Path(CONFIG_FILE).exists():
                with open(CONFIG_FILE, 'r') as f:
                    speakers = json.load(f)
                    speaker_map = {speaker['name']: speaker['slug'] for speaker in speakers}
            
            # Add slug field to timeline segments and round timing
            for segment in timeline:
                speaker_name = segment['speaker']
                segment['slug'] = speaker_map.get(speaker_name, speaker_name.lower())
                # Round timing to 2 decimal places for cleaner processing
                segment['start'] = round(segment['start'], 2)
                segment['end'] = round(segment['end'], 2)
            
            print(f"✓ Loaded speaker timeline with {len(timeline)} segments")
            for i, seg in enumerate(timeline):
                duration = seg['end'] - seg['start']
                print(f"  {i+1}. {seg['speaker']:8s}: {seg['start']:6.2f}-{seg['end']:6.2f}s ({duration:5.2f}s)")
            
            return timeline
            
        except Exception as e:
            print(f"Error loading speaker timeline: {e}")
            return []
    
    def _create_speaker_clips(self, speakers: List[Dict], duration: float) -> Dict[str, str]:
        """Create standardized video clips for each speaker."""
        video_clips = {}
        
        print("Creating speaker video clips...")
        
        for speaker in speakers:
            slug = speaker['slug']
            name = speaker['name']
            image_path = Path(IMAGES_DIR) / f"{slug}.jpg"
            clip_path = Path(self.temp_dir) / f"{slug}.mp4"
            
            if image_path.exists():
                # Create video from image
                cmd = [
                    'ffmpeg', '-y', '-loop', '1', '-i', str(image_path),
                    '-c:v', 'libx264', '-t', '1', '-pix_fmt', 'yuv420p',
                    '-vf', f'scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,'
                           f'pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1',
                    '-loglevel', 'error', str(clip_path)
                ]
            else:
                print(f"Warning: Image not found for {name}, using text placeholder")
                # Create text placeholder
                safe_name = name.replace("'", "\\'").replace(":", "\\:")
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi', 
                    '-i', f'color=c=black:s={self.width}x{self.height}:r=25:d=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', f'drawtext=text=\'{safe_name}\':fontcolor=white:fontsize=48:'
                           f'x=(w-text_w)/2:y=(h-text_h)/2,setsar=1',
                    '-loglevel', 'error', str(clip_path)
                ]
            
            try:
                subprocess.run(cmd, check=True)
                video_clips[slug] = str(clip_path)
                print(f"✓ Created clip for {name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to create clip for {name}: {e}")
                return {}
        
        return video_clips
    
    
    def _assemble_final_video(
        self,
        video_clips: Dict[str, str],
        timeline: List[Dict],
        audio_path: str,
        output_path: str,
        enable_subtitles: bool,
        srt_path: str
    ) -> bool:
        """Assemble the final video with speaker switching."""
        if not timeline:
            print("No timeline data, creating fallback video")
            # Use first speaker for entire duration
            first_slug = list(video_clips.keys())[0]
            return self._create_single_speaker_video(
                video_clips[first_slug], 
                audio_path, 
                output_path,
                enable_subtitles,
                srt_path
            )
        
        print("Assembling final video with speaker switching...")
        
        # Use the speaker timeline directly - it already has the correct timing
        full_timeline = timeline
        
        # Build FFmpeg filter complex
        input_files = []
        slugs = list(video_clips.keys())
        
        for slug, clip_path in video_clips.items():
            input_files.extend(['-i', clip_path])
        
        # Add audio input
        input_files.extend(['-i', audio_path])
        audio_index = len(video_clips)
        
        # Create video segments for each time period
        filter_parts = []
        video_segments = []
        segment_index = 0
        
        for segment in full_timeline:
            speaker_slug = segment['slug']
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            if duration > 0.01:  # Skip very short segments
                try:
                    speaker_index = slugs.index(speaker_slug)
                except ValueError:
                    # Fallback to first speaker if slug not found
                    speaker_index = 0
                
                # Create a video segment by looping the speaker's clip for the required duration
                filter_parts.append(
                    f'[{speaker_index}:v]loop=loop=-1:size=1:start=0,trim=duration={duration:.3f},setpts=PTS-STARTPTS[v{segment_index}]'
                )
                video_segments.append(f'[v{segment_index}]')
                segment_index += 1
        
        if not video_segments:
            print("No video segments created")
            return False
        
        # Concatenate all segments
        concat_inputs = ''.join(video_segments)
        filter_parts.append(
            f'{concat_inputs}concat=n={len(video_segments)}:v=1:a=0[v_concat]'
        )
        
        # Pad for even dimensions
        filter_parts.append('[v_concat]pad=ceil(iw/2)*2:ceil(ih/2)*2[v_padded]')
        output_stream = '[v_padded]'
        
        # Add subtitles if enabled
        if enable_subtitles and Path(srt_path).exists():
            style = "Alignment=2,Fontsize=18,PrimaryColour=&HFFFFFF&,BorderStyle=1,Outline=1,Shadow=1,MarginV=25"
            filter_parts.append(
                f'{output_stream}subtitles=\'{srt_path}\':force_style=\'{style}\'[v_final]'
            )
            output_stream = '[v_final]'
        
        filter_complex = ';'.join(filter_parts)
        
        # Build final FFmpeg command
        cmd = [
            'ffmpeg', '-y'
        ] + input_files + [
            '-filter_complex', filter_complex,
            '-map', output_stream,
            '-map', f'{audio_index}:a',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
            '-pix_fmt', 'yuv420p', '-r', '25',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest', '-loglevel', 'warning',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Video generated successfully: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed: {e}")
            return False
    
    def _create_single_speaker_video(
        self, 
        video_clip: str, 
        audio_path: str, 
        output_path: str,
        enable_subtitles: bool,
        srt_path: str
    ) -> bool:
        """Create video with single speaker (fallback)."""
        print("Creating single-speaker video...")
        
        filter_parts = []
        output_stream = '[0:v]'
        
        if enable_subtitles and Path(srt_path).exists():
            style = "Alignment=2,Fontsize=18,PrimaryColour=&HFFFFFF&,BorderStyle=1,Outline=1,Shadow=1,MarginV=25"
            filter_parts.append(f'[0:v]subtitles=\'{srt_path}\':force_style=\'{style}\'[v_final]')
            output_stream = '[v_final]'
        
        cmd = [
            'ffmpeg', '-y', '-stream_loop', '-1', '-i', video_clip, '-i', audio_path,
            '-map', output_stream, '-map', '1:a',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest', '-loglevel', 'warning',
            output_path
        ]
        
        if filter_parts:
            cmd.insert(-8, '-filter_complex')
            cmd.insert(-8, ';'.join(filter_parts))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Single-speaker video generated: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to create single-speaker video: {e}")
            return False