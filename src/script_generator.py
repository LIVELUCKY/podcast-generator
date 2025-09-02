"""Script generation using AI models."""

import os
import re
import asyncio
import aiofiles
from pathlib import Path
import google.generativeai as genai
import nltk

from .config import (WORDS_PER_MINUTE, SCRIPT_CACHE_FILE, GEMINI_MODEL, FUZZY_MATCH_THRESHOLD,
                     SCRIPT_LENGTH_TOLERANCE, SCRIPT_MAX_LENGTH_MULTIPLIER, CHUNK_SIZE_WORDS)

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        try:
            nltk.download('punkt')
        except:
            pass  # Fall back to simple word counting


def clean_gemini_output(raw_text: str) -> str:
    """Clean Gemini output to extract script dialogue."""
    code_block_pattern = re.compile(r"```(?:text|script)?\n(.*?)\n```", re.DOTALL)
    speaker_pattern = re.compile(
        r"^(Speaker\s+\d+:\s*.*)", re.MULTILINE | re.IGNORECASE
    )

    cleaned_text = code_block_pattern.sub(r"\1", raw_text).strip()
    matches = speaker_pattern.findall(cleaned_text)

    return "\n".join(line.strip() for line in matches) if matches else ""


class ScriptGenerator:
    """Handles script generation and caching."""
    
    def __init__(self, speaker_voices):
        self.speaker_voices = speaker_voices
    
    async def generate_script(self, topic: str, duration: int, force_regenerate: bool = False) -> str:
        """Generate or load cached script."""
        script_path = Path(SCRIPT_CACHE_FILE)
        if script_path.exists() and not force_regenerate:
            async with aiofiles.open(script_path, "r", encoding="utf-8") as f:
                existing_script = await f.read()
            
            if self._validate_script_quality(existing_script, duration):
                return existing_script
            else:
                print("📝 Existing script fails quality validation, regenerating...")
                force_regenerate = True
        
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"🔄 Generating script (attempt {attempt + 1}/{max_attempts})...")
            script_text = await self._generate_script_in_chunks(topic, duration, attempt + 1)
            
            if self._validate_script_quality(script_text, duration):
                async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
                    await f.write(script_text)
                print("✅ Script generation successful!")
                return script_text
            else:
                print(f"⚠️  Script attempt {attempt + 1} failed validation")
                if attempt < max_attempts - 1:
                    print("🔄 Retrying with enhanced prompt...")
        
        raise ValueError("Failed to generate a quality script after maximum attempts")
    
    async def _generate_script_in_chunks(self, topic: str, duration: int, attempt: int = 1) -> str:
        """Generate script in multiple chunks to handle LLM output length limitations."""
        target_word_count = duration * WORDS_PER_MINUTE
        
        # Adaptive chunking based on target word count and configurable chunk size
        if target_word_count > CHUNK_SIZE_WORDS:
            # Calculate optimal chunk size and count for any duration
            words_per_chunk = CHUNK_SIZE_WORDS
            chunks_needed = max(2, int((target_word_count + words_per_chunk - 1) // words_per_chunk))
            
            # Distribute duration evenly across chunks
            chunk_duration_minutes = duration / chunks_needed
            
            # Ensure minimum chunk duration of 0.5 minutes
            if chunk_duration_minutes < 0.5:
                chunk_duration_minutes = 0.5
                chunks_needed = max(1, int(duration / chunk_duration_minutes))
            
            print(f"📝 Generating {duration}-minute script in {chunks_needed} chunks...")
            
            script_parts = []
            remaining_duration = duration
            
            for chunk_idx in range(chunks_needed):
                current_chunk_duration = min(chunk_duration_minutes, remaining_duration)
                if chunk_idx == chunks_needed - 1:  # Last chunk gets any remaining time
                    current_chunk_duration = remaining_duration
                
                print(f"   🔄 Generating chunk {chunk_idx + 1}/{chunks_needed} ({current_chunk_duration} min)...")
                
                if chunk_idx == 0:
                    # First chunk: Introduction and opening
                    chunk_script = await self._generate_script_chunk(
                        topic, current_chunk_duration, "opening",
                        previous_content="", chunk_idx=chunk_idx, total_chunks=chunks_needed
                    )
                elif chunk_idx == chunks_needed - 1:
                    # Last chunk: Conclusion
                    chunk_script = await self._generate_script_chunk(
                        topic, current_chunk_duration, "conclusion",
                        previous_content="\n".join(script_parts), chunk_idx=chunk_idx, total_chunks=chunks_needed
                    )
                else:
                    # Middle chunk: Continuation
                    chunk_script = await self._generate_script_chunk(
                        topic, current_chunk_duration, "continuation",
                        previous_content="\n".join(script_parts), chunk_idx=chunk_idx, total_chunks=chunks_needed
                    )
                
                if chunk_script:
                    script_parts.append(chunk_script)
                    remaining_duration -= current_chunk_duration
                else:
                    raise ValueError(f"Failed to generate chunk {chunk_idx + 1}")
            
            # Combine all chunks and ensure smooth transitions
            combined_script = self._combine_script_chunks(script_parts)
            return combined_script
        else:
            # For shorter scripts, generate in one go
            return await self._generate_new_script(topic, duration, attempt)
    
    async def _generate_script_chunk(self, topic: str, chunk_duration: int, chunk_type: str, 
                                   previous_content: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate a specific chunk of the script."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        available_characters = [
            (name, info) for name, info in self.speaker_voices.items() 
            if info.get("has_voice_sample")
        ]
        
        speaker_assignments = "\n".join(
            f"- Speaker {i+1}: {name} ({info['persona']})"
            for i, (name, info) in enumerate(available_characters)
        )
        
        target_word_count = chunk_duration * WORDS_PER_MINUTE
        
        # Create context from previous content
        context_prompt = ""
        if previous_content:
            last_lines = previous_content.strip().split('\n')[-3:]  # Get last 3 lines for context
            context_prompt = f"""
PREVIOUS CONVERSATION CONTEXT:
{chr(10).join(last_lines)}

CONTINUE THE CONVERSATION naturally from this point. The next speaker should respond to what was just said.
"""
        
        # Chunk-specific instructions
        if chunk_type == "opening":
            chunk_instructions = f"""
This is the OPENING chunk (part {chunk_idx + 1} of {total_chunks}) of a {chunk_duration}-minute segment.
- Start with Speaker 1 introducing the topic and participants
- Establish the conversation tone and direction  
- End this chunk at a natural conversation break point
- Ensure speakers are engaging with each other, not giving monologues
"""
        elif chunk_type == "conclusion":
            chunk_instructions = f"""
This is the CONCLUSION chunk (part {chunk_idx + 1} of {total_chunks}) of a {chunk_duration}-minute segment.
- Continue naturally from the previous conversation
- Begin wrapping up the discussion
- Have Speaker 1 summarize key points discussed throughout the entire conversation
- End with a satisfying conclusion
"""
        else:
            chunk_instructions = f"""
This is a MIDDLE chunk (part {chunk_idx + 1} of {total_chunks}) of a {chunk_duration}-minute segment.
- Continue the conversation naturally from where it left off
- Develop the topic deeper with more detailed exchanges
- End at a natural pause point that allows smooth continuation
- Maintain conversation flow and speaker engagement
"""
        
        prompt = f"""Continue writing a podcast script chunk on "{topic}".
Chunk duration: {chunk_duration} minutes (~{target_word_count} words).
{context_prompt}
SPEAKERS:
{speaker_assignments}

{chunk_instructions}

CRITICAL RULES:
1. Continue conversation naturally - speakers must respond to each other
2. Each line MUST start with "Speaker X: " where X is 1, 2, 3, etc.
3. Output ONLY dialogue, no stage directions
4. Each response should be 2-4 sentences
5. Maintain natural conversation flow with references to previous statements
6. Aim for approximately {target_word_count} words in this chunk

Begin chunk:"""
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = await asyncio.to_thread(model.generate_content, prompt)
        return clean_gemini_output(response.text)
    
    def _combine_script_chunks(self, chunks: list) -> str:
        """Combine script chunks ensuring smooth transitions."""
        if not chunks:
            return ""
        
        combined = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            chunk_lines = [line.strip() for line in chunk.strip().split('\n') if line.strip()]
            
            if i == 0:
                # First chunk - add as is
                combined.extend(chunk_lines)
            else:
                # Subsequent chunks - check for smooth transition
                if combined and chunk_lines:
                    last_speaker = None
                    first_speaker = None
                    
                    # Get last speaker from previous chunk
                    for line in reversed(combined):
                        if re.match(r'^Speaker \d+:', line):
                            last_speaker = re.match(r'^Speaker (\d+):', line).group(1)
                            break
                    
                    # Get first speaker from current chunk
                    for line in chunk_lines:
                        if re.match(r'^Speaker \d+:', line):
                            first_speaker = re.match(r'^Speaker (\d+):', line).group(1)
                            break
                    
                    # If same speaker continues, merge or add transition
                    if last_speaker == first_speaker and chunk_lines:
                        # Same speaker continuing - merge content if it flows naturally
                        combined.extend(chunk_lines)
                    else:
                        combined.extend(chunk_lines)
        
        return '\n'.join(combined)
    
    async def _generate_new_script(self, topic: str, duration: int, attempt: int = 1) -> str:
        """Generate new script using Gemini API with enhanced prompt for better quality."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)

        available_characters = [
            (name, info) for name, info in self.speaker_voices.items() 
            if info.get("has_voice_sample")
        ]
        if not available_characters:
            raise ValueError("No characters with voice samples are configured.")

        speaker_assignments = "\n".join(
            f"- Speaker {i+1}: {name} ({info['persona']})"
            for i, (name, info) in enumerate(available_characters)
        )
        
        target_word_count = duration * WORDS_PER_MINUTE
        min_words = int(target_word_count * SCRIPT_LENGTH_TOLERANCE)
        max_words = int(target_word_count * SCRIPT_MAX_LENGTH_MULTIPLIER)
        
        # Enhanced prompt with AI correctness validation
        quality_emphasis = ""
        if attempt > 1:
            quality_emphasis = f"""
CRITICAL: This is attempt {attempt}. Previous attempts failed quality validation.
FOCUS ESPECIALLY ON:
- Meeting the flexible word count requirement (aim for {target_word_count}, acceptable: {min_words}-{max_words} words)
- Creating fluid conversation where speakers directly reference each other
- Ensuring ALL speakers have balanced participation
- Using natural conversation transitions and responses
"""
        
        prompt = f"""Write a complete, high-quality podcast script on "{topic}".
Duration: {duration} minutes (TARGET: {target_word_count} words, MINIMUM: {min_words} words, MAXIMUM: {max_words} words).
{quality_emphasis}
FLEXIBLE LENGTH REQUIREMENT:
- IDEAL: Aim for approximately {target_word_count} words ({duration} minutes at {WORDS_PER_MINUTE} WPM)
- ACCEPTABLE RANGE: {min_words} to {max_words} words ({SCRIPT_LENGTH_TOLERANCE:.0%}-{SCRIPT_MAX_LENGTH_MULTIPLIER:.0%} of target)
- Each speaker should have roughly equal speaking time
- Focus on content quality over exact word count
- Natural conversation flow is more important than hitting precise numbers

MANDATORY: You MUST use ALL {len(available_characters)} speakers in the conversation:
{speaker_assignments}

CRITICAL CONVERSATION FLOW RULES:
1. Speakers must DIRECTLY respond to each other's questions and points
2. When one speaker asks a question, the next speaker must answer THAT specific question
3. Build on previous speakers' ideas - reference what they just said
4. Use natural conversation connectors like "That's interesting, but...", "Building on what you said...", "I agree with your point about..."
5. Create genuine dialogue exchanges, not parallel monologues
6. Each response should be 3-5 sentences to ensure adequate content length
7. Include words from previous speakers' responses to show active listening

AI CORRECTNESS VALIDATION REQUIREMENTS:
- Stay factually accurate and avoid making claims that can't be substantiated
- If discussing technical topics, ensure explanations are clear and correct
- Maintain consistency in speaker personalities and expertise areas
- Avoid contradictory statements between speakers unless it's intentional disagreement
- Use appropriate terminology for the topic without being overly technical
- Ensure logical flow of ideas throughout the conversation

Podcast Structure:
1. Speaker 1: Brief introduction and topic setup (2-3 sentences)
2. Speaker 1: Ask Speaker 2 a SPECIFIC detailed question about their expertise
3. Speaker 2: Answer the question thoroughly (4-5 sentences), then ask Speaker 3 something related
4. Speaker 3: Respond to Speaker 2's question comprehensively, build on their point, then engage Speaker 4
5. Continue this pattern - each speaker responds substantively AND asks/engages the next
6. Include moments where speakers disagree, ask follow-up questions, or challenge each other
7. Add deeper dives into subtopics to reach target word count
8. End with Speaker 1 summarizing key points from the discussion (3-4 sentences)

Conversation Examples:
✓ GOOD: "Speaker 2, you mentioned AI can see patterns we can't. Can you give us a concrete example and explain how this impacts daily business decisions?" followed by Speaker 2 giving detailed examples
✓ GOOD: "That's fascinating, Morgan. But Anthony, don't you think there are risks to relying too heavily on these algorithms? What safeguards should companies implement?"
✓ GOOD: "Building on what Sarah just said about machine learning algorithms, I think we need to consider the human element too..."
✗ BAD: Speakers making statements that ignore what others just said
✗ BAD: Questions that aren't answered by the next speaker
✗ BAD: Short, superficial responses that don't contribute to the target word count

Technical Requirements:
1. Each line MUST start with "Speaker X: " where X matches the assignments above
2. ALL {len(available_characters)} speakers must appear multiple times throughout
3. Output ONLY the dialogue. No stage directions, sound effects, or notes
4. Each response should be 3-5 sentences to ensure proper length
5. Every speaker transition should feel like a natural conversation flow
6. Aim for the target word count but prioritize natural flow over exact numbers
7. Double-check your work for factual accuracy and logical consistency

Begin script:"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = await asyncio.to_thread(model.generate_content, prompt)
        return clean_gemini_output(response.text)
    
    def _has_good_conversation_flow(self, script: str) -> bool:
        """Check if script has good conversational flow between speakers."""
        lines = [line.strip() for line in script.strip().split('\n') if line.strip()]
        
        if len(lines) < 5:
            return False
        
        conversation_markers = [
            r'\?',
            r'(that\'s|you mentioned|you said|i agree|building on|interesting|but|however)',
            r'(you|your)',
            r'(what do you think|how do you see|can you|would you)',
        ]
        
        marker_count = 0
        question_response_pairs = 0
        
        for i, line in enumerate(lines):
            for marker in conversation_markers:
                if re.search(marker, line, re.IGNORECASE):
                    marker_count += 1
                    break
            
            if '?' in line and i + 1 < len(lines):
                current_speaker = re.match(r'^Speaker (\d+):', line)
                next_speaker = re.match(r'^Speaker (\d+):', lines[i + 1])
                if current_speaker and next_speaker and current_speaker.group(1) != next_speaker.group(1):
                    question_response_pairs += 1
        
        marker_ratio = marker_count / len(lines) if lines else 0
        has_good_flow = marker_ratio >= 0.3 and question_response_pairs >= 2
        
        if not has_good_flow:
            print(f"📊 Conversation flow analysis:")
            print(f"   - Conversation markers: {marker_count}/{len(lines)} ({marker_ratio:.1%})")
            print(f"   - Question-response pairs: {question_response_pairs}")
            print(f"   - Flow quality: {'Good' if has_good_flow else 'Poor'}")
        
        return has_good_flow
    
    def _validate_script_quality(self, script: str, target_duration: int) -> bool:
        """Comprehensive script quality validation."""
        if not script or not script.strip():
            print("❌ Script is empty")
            return False
        
        # Check word count with configurable tolerance
        word_count = self._count_words(script)
        target_words = target_duration * WORDS_PER_MINUTE
        min_words = int(target_words * SCRIPT_LENGTH_TOLERANCE)
        max_words = int(target_words * SCRIPT_MAX_LENGTH_MULTIPLIER)
        
        print(f"📊 Script validation:")
        print(f"   - Word count: {word_count} (target: {target_words}, acceptable: {min_words}-{max_words})")
        print(f"   - Tolerance: {SCRIPT_LENGTH_TOLERANCE:.0%} minimum, {SCRIPT_MAX_LENGTH_MULTIPLIER:.0%} maximum")
        
        if word_count < min_words:
            print(f"❌ Script too short: {word_count} < {min_words} words ({word_count/target_words:.1%} of target)")
            return False
        
        if word_count > max_words:
            print(f"❌ Script too long: {word_count} > {max_words} words ({word_count/target_words:.1%} of target)")
            return False
        
        # Show success percentage
        percentage = (word_count / target_words) * 100
        print(f"   ✅ Length acceptable: {percentage:.1f}% of target")
        
        # Check conversation flow
        if not self._has_good_conversation_flow(script):
            print("❌ Script lacks proper conversation flow")
            return False
        
        # Check speaker distribution
        if not self._validate_speaker_distribution(script):
            print("❌ Script has poor speaker distribution")
            return False
        
        # Check content fluidity using fuzzy matching
        if not self._validate_content_fluidity(script):
            print("❌ Script content lacks fluidity")
            return False
        
        print("✅ Script passes all quality checks")
        return True
    
    def _count_words(self, text: str) -> int:
        """Count words in text using NLTK tokenization or fallback."""
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text.lower())
            # Filter out punctuation and count only actual words
            words = [token for token in tokens if token.isalnum()]
            return len(words)
        except Exception:
            # Fallback to simple splitting if NLTK fails
            # Remove common punctuation and count words
            cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
            words = [word for word in cleaned.split() if word.strip()]
            return len(words)
    
    def _validate_speaker_distribution(self, script: str) -> bool:
        """Check if all speakers are used fairly."""
        lines = [line.strip() for line in script.strip().split('\n') if line.strip()]
        speaker_counts = {}
        
        for line in lines:
            speaker_match = re.match(r'^Speaker (\d+):', line)
            if speaker_match:
                speaker_num = speaker_match.group(1)
                speaker_counts[speaker_num] = speaker_counts.get(speaker_num, 0) + 1
        
        if not speaker_counts:
            return False
        
        # Check if all configured speakers are used
        available_speakers = len([name for name, info in self.speaker_voices.items() 
                                if info.get("has_voice_sample")])
        
        expected_speakers = set(str(i+1) for i in range(available_speakers))
        actual_speakers = set(speaker_counts.keys())
        
        if expected_speakers != actual_speakers:
            print(f"   - Missing speakers: {expected_speakers - actual_speakers}")
            return False
        
        # Check distribution fairness (no speaker should have less than 60% of the average)
        avg_lines = sum(speaker_counts.values()) / len(speaker_counts)
        min_threshold = avg_lines * 0.6
        
        for speaker, count in speaker_counts.items():
            if count < min_threshold:
                print(f"   - Speaker {speaker} underused: {count} vs avg {avg_lines:.1f}")
                return False
        
        return True
    
    def _validate_content_fluidity(self, script: str) -> bool:
        """Check content fluidity using fuzzy string matching for conversation continuity."""
        lines = [line.strip() for line in script.strip().split('\n') if line.strip()]
        
        if len(lines) < 3:
            return False
        
        fluidity_score = 0
        total_transitions = 0
        
        # Check for conversation continuity markers
        continuity_patterns = [
            r'\b(that\'s|you mentioned|you said|as you noted|building on)\b',
            r'\b(i agree|disagree|however|but|although|though)\b',
            r'\b(can you|would you|how do you|what do you think)\b',
            r'\b(exactly|precisely|absolutely|definitely|certainly)\b'
        ]
        
        for i in range(len(lines) - 1):
            current_line = lines[i].lower()
            next_line = lines[i + 1].lower()
            
            # Check for direct responses using word overlap
            try:
                from nltk.tokenize import word_tokenize
                current_words = set(word_tokenize(current_line))
                next_words = set(word_tokenize(next_line))
            except:
                # Fallback to simple tokenization
                current_words = set(re.findall(r'\b\w+\b', current_line.lower()))
                next_words = set(re.findall(r'\b\w+\b', next_line.lower()))
            
            # Calculate word overlap
            common_words = current_words.intersection(next_words)
            if len(common_words) > 0:
                overlap_ratio = len(common_words) / max(len(current_words), len(next_words))
                if overlap_ratio >= 0.1:  # At least 10% word overlap
                    fluidity_score += 1
            
            # Check for conversation continuity markers
            for pattern in continuity_patterns:
                if re.search(pattern, next_line):
                    fluidity_score += 1
                    break
            
            total_transitions += 1
        
        fluidity_ratio = fluidity_score / total_transitions if total_transitions > 0 else 0
        required_fluidity = FUZZY_MATCH_THRESHOLD * 0.7  # 70% of fuzzy threshold
        
        print(f"   - Content fluidity: {fluidity_ratio:.2f} (required: {required_fluidity:.2f})")
        
        return fluidity_ratio >= required_fluidity