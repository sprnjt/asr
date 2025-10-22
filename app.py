import re
import os
import json
import sys
import numpy as np
from typing import List, Dict
from pydantic import BaseModel
import librosa
import soundfile as sf
from sarvamai import SarvamAI
import jiwer
import google.generativeai as genai
from dotenv import load_dotenv

class Segment(BaseModel):
    start_ts: float
    end_ts: float
    transcript: str
    speaker: str = ""
    translation: str = ""

def count_syllables(text: str) -> int:
    text = text.lower()
    vowels = "aeiou"
    syllable_count = 0
    previous_was_vowel = False
    
    for char in text:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    if text.endswith('e'):
        syllable_count -= 1
    
    return max(1, syllable_count)

class SarvamBatchTranscriber:
    def __init__(self, api_key: str):
        self.client = SarvamAI(api_subscription_key=api_key)
    
    def transcribe_batch(
        self, 
        audio_paths: List[str], 
        language: str = "hi-IN",
        num_speakers: int = 2,
        output_dir: str = "./output"
    ) -> List[Dict]:
        print("\n[Batch API] Creating transcription job...")
        
        job = self.client.speech_to_text_job.create_job(
            language_code=language,
            model="saarika:v2.5",
            with_timestamps=True,
            with_diarization=True,
            num_speakers=num_speakers
        )
        
        print(f"✓ Job created with ID: {job.job_id}")
        
        print(f"\n[Batch API] Uploading {len(audio_paths)} file(s)...")
        job.upload_files(file_paths=audio_paths)
        print("✓ Files uploaded")
        
        print("\n[Batch API] Starting transcription...")
        job.start()
        
        print("[Batch API] Waiting for job to complete...")
        final_status = job.wait_until_complete()
        
        if job.is_failed():
            raise Exception("STT job failed")
        
        print(f"✓ Job completed with status: {final_status}")
        
        print(f"\n[Batch API] Downloading outputs to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        job.download_outputs(output_dir=output_dir)
        print("✓ Outputs downloaded")
        
        return self._parse_batch_results(output_dir)
    
    def _parse_batch_results(self, output_dir: str) -> List[Dict]:
        segments = []
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'segments' in data:
                    for seg in data['segments']:
                        segments.append({
                            'start_ts': seg.get('start', 0.0),
                            'end_ts': seg.get('end', 0.0),
                            'transcript': seg.get('text', ''),
                            'speaker': seg.get('speaker', 'SPEAKER_00')
                        })
                elif 'transcript' in data:
                    segments.append({
                        'start_ts': 0.0,
                        'end_ts': 0.0,
                        'transcript': data['transcript'],
                        'speaker': 'SPEAKER_00'
                    })
        
        return segments
    
    def transcribe_segment(
        self, 
        audio_path: str, 
        start: float, 
        end: float, 
        language: str = "hi-IN"
    ) -> str:
        audio, sr = librosa.load(audio_path, sr=16000)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]
        
        temp_path = "/tmp/segment_audio.wav"
        sf.write(temp_path, segment_audio, sr)
        
        segments = self.transcribe_batch(
            [temp_path], 
            language=language,
            num_speakers=1,
            output_dir="/tmp/segment_output"
        )
        
        if segments:
            return segments[0]['transcript']
        return ""

def detect_compound_proper_nouns(text: str) -> List[Dict[str, any]]:
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    matches = re.finditer(pattern, text)
    
    proper_nouns = []
    for match in matches:
        noun = match.group(0)
        syllables = count_syllables(noun)
        
        words = noun.split()
        variants = [
            noun,
            ' '.join(words[:2]),
            ''.join([w[0] for w in words]),
        ]
        
        proper_nouns.append({
            'text': noun,
            'syllables': syllables,
            'start_pos': match.start(),
            'end_pos': match.end(),
            'variants': variants
        })
    
    return proper_nouns

def find_proper_nouns_in_segments(segments: List[Dict]) -> List[Dict]:
    proper_nouns_timed = []
    
    for seg in segments:
        text = seg['transcript']
        nouns = detect_compound_proper_nouns(text)
        
        for noun in nouns:
            char_ratio = noun['start_pos'] / len(text) if len(text) > 0 else 0
            duration = seg['end_ts'] - seg['start_ts']
            
            proper_nouns_timed.append({
                'text': noun['text'],
                'syllables': noun['syllables'],
                'variants': noun['variants'],
                'start_ts': seg['start_ts'] + (char_ratio * duration),
                'end_ts': seg['start_ts'] + (char_ratio * duration) + 
                            (noun['syllables'] / count_syllables(text) * duration),
                'segment_idx': segments.index(seg),
                'speaker': seg.get('speaker', 'UNKNOWN')
            })
    
    return proper_nouns_timed

def llm_trie_manager(full_text: str, context: str, gemini_api_key: str = None) -> str:
    if gemini_api_key is None:
        return full_text
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    words = full_text.split()
    chunk_size = 200
    overlap = 20
    corrected_chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        
        prompt = f"""Context: {context}

Original transcript (may have ASR errors):
{chunk}

Fix any obvious transcription errors while preserving the meaning. Only correct clear mistakes like:
- Wrong proper nouns (e.g., "Indian Oil Cooperation" → "Indian Oil Corporation")
- Homophones (e.g., "there" vs "their")
- Common ASR errors

Return ONLY the corrected text, nothing else:"""
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.99,
                    max_output_tokens=500,
                )
            )
            corrected = response.text.strip()
            corrected_chunks.append(corrected)
            print(f"   ✓ Processed chunk {i//chunk_size + 1}")
        except Exception as e:
            print(f"   ✗ Gemini error: {e}")
            corrected_chunks.append(chunk)
    
    return ' '.join(corrected_chunks)

def algorithm(
    audio_path: str,
    sarvam_api_key: str,
    gemini_api_key: str = None,
    language: str = "hi-IN",
    context: str = "Business/corporate audio translation",
    num_speakers: int = 2,
    use_llm: bool = True
):
    print("=" * 80)
    print("PROPER NOUN CORRECTION PIPELINE (Sarvam Batch API + Gemini)")
    print("=" * 80)
    
    transcriber = SarvamBatchTranscriber(sarvam_api_key)
    
    print("\n[1/5] Transcribing with Sarvam Batch API (timestamps + diarization)...")
    segments = transcriber.transcribe_batch(
        audio_paths=[audio_path],
        language=language,
        num_speakers=num_speakers,
        output_dir="./batch_output"
    )
    
    full_transcript = ' '.join([seg['transcript'] for seg in segments])
    
    print(f"✓ Transcript length: {len(full_transcript)} chars")
    print(f"✓ Word count: {len(full_transcript.split())} words")
    print(f"✓ Segments: {len(segments)}")
    print(f"Preview: {full_transcript[:200]}...")
    
    print("\n[2/5] Calculating speaking rate...")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    total_syllables = count_syllables(full_transcript)
    spm = (total_syllables / duration) * 60 if duration > 0 else 0
    print(f"✓ Duration: {duration:.1f}s")
    print(f"✓ Syllables: {total_syllables}")
    print(f"✓ Speaking rate: {spm:.1f} syllables/minute")
    
    print("\n[3/5] Detecting proper nouns with timestamps...")
    proper_nouns_timed = find_proper_nouns_in_segments(segments)
    print(f"✓ Found {len(proper_nouns_timed)} proper nouns with timestamps:")
    for pn in proper_nouns_timed[:5]:
        print(f"   - {pn['text']} ({pn['start_ts']:.2f}s-{pn['end_ts']:.2f}s, {pn['speaker']})")
    if len(proper_nouns_timed) > 5:
        print(f"   ... and {len(proper_nouns_timed) - 5} more")
    
    print("\n[4/5] Re-translating proper noun segments...")
    corrected_transcript = full_transcript
    
    for idx, noun_info in enumerate(proper_nouns_timed[:10], 1):
        print(f"\n   [{idx}/{min(10, len(proper_nouns_timed))}] Correcting: {noun_info['text']}")
        print(f"       Time: {noun_info['start_ts']:.2f}s - {noun_info['end_ts']:.2f}s")
        print(f"       Speaker: {noun_info['speaker']}")
        
        try:
            better_translation = transcriber.transcribe_segment(
                audio_path,
                noun_info['start_ts'],
                noun_info['end_ts'],
                language="en-IN"
            )
            
            corrected_transcript = corrected_transcript.replace(
                noun_info['text'],
                better_translation,
                1
            )
            print(f"       ✓ Replaced with: {better_translation}")
        except Exception as e:
            print(f"       ✗ Error: {e}")
            continue
    
    if use_llm and gemini_api_key:
        print("\n[5/5] Applying Gemini LLM corrections...")
        try:
            corrected_transcript = llm_trie_manager(
                corrected_transcript,
                context,
                gemini_api_key
            )
            print("✓ Gemini corrections applied")
        except Exception as e:
            print(f"✗ Gemini correction failed: {e}")
    else:
        print("\n[5/5] Skipping LLM corrections")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nOriginal (first 500 chars):\n{full_transcript[:500]}...")
    print(f"\nCorrected (first 500 chars):\n{corrected_transcript[:500]}...")
    
    chars_changed = sum(c1 != c2 for c1, c2 in zip(full_transcript, corrected_transcript))
    print(f"\nChanges made: {chars_changed} characters")
    print(f"Proper nouns processed: {min(10, len(proper_nouns_timed))}")
    
    return corrected_transcript, segments

def evaluate_quality(original: str, corrected: str, reference: str = None) -> Dict:
    results = {
        'original_length': len(original),
        'corrected_length': len(corrected),
        'chars_changed': sum(c1 != c2 for c1, c2 in zip(original, corrected)),
        'change_percentage': (sum(c1 != c2 for c1, c2 in zip(original, corrected)) / len(original) * 100) if len(original) > 0 else 0
    }
    
    if reference:
        results['original_wer'] = wer(reference, original)
        results['corrected_wer'] = wer(reference, corrected)
        results['wer_improvement'] = results['original_wer'] - results['corrected_wer']
        results['improvement_percentage'] = (results['wer_improvement'] / results['original_wer'] * 100) if results['original_wer'] > 0 else 0
    
    return results

if __name__ == "__main__":
    load_dotenv()
    
    AUDIO_FILE = "10minpod.mp3"
    SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    if not SARVAM_API_KEY:
        print("Error: SARVAM_API_KEY not found in environment variables.", file=sys.stderr)
        print("Please create a .env file and add: SARVAM_API_KEY=your_key", file=sys.stderr)
        sys.exit(1)
    
    use_llm_flag = True
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found. LLM corrections (Step 5) will be skipped.")
        use_llm_flag = False
    
    corrected_transcript, segments = algorithm(
        audio_path=AUDIO_FILE,
        sarvam_api_key=SARVAM_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        language="hi-IN",
        context="Corporate podcast about business, stock market, and Indian companies",
        num_speakers=2,
        use_llm=use_llm_flag
    )
    
    output_file = "corrected_transcript.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(corrected_transcript)
    print(f"\n✓ Final transcript saved to: {output_file}")
    
    segments_file = "segments_with_timestamps.json"
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"✓ Segments saved to: {segments_file}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)