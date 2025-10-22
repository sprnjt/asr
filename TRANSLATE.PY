import re
import os
import json
import time
import numpy as np
from typing import List, Dict
from pydantic import BaseModel
import librosa
import soundfile as sf
from sarvamai import SarvamAI
from jiwer import wer
import google.generativeai as genai
from dotenv import load_dotenv

class Segment(BaseModel):
    start_ts: float
    end_ts: float
    transcript: str
    speaker: str = ""
    translation: str = ""

class RateLimiter:
    def __init__(self, calls_per_minute: int = 15):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            print(f"  [Rate Limit] Waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

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
                
                print(f"\n[Parser] Reading: {filename}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"[Parser] JSON keys: {data.keys()}")
                
                segment_data = None
                
                if 'segments' in data:
                    segment_data = data['segments']
                elif 'results' in data and isinstance(data['results'], list):
                    for result in data['results']:
                        if 'segments' in result:
                            segment_data = result['segments']
                            break
                elif 'data' in data and 'segments' in data['data']:
                    segment_data = data['data']['segments']
                
                if segment_data:
                    print(f"[Parser] Found {len(segment_data)} segments")
                    for seg in segment_data:
                        start = seg.get('start', seg.get('start_time', seg.get('start_ts', 0.0)))
                        end = seg.get('end', seg.get('end_time', seg.get('end_ts', 0.0)))
                        text = seg.get('text', seg.get('transcript', ''))
                        speaker = seg.get('speaker', seg.get('speaker_id', seg.get('speaker_label', 'SPEAKER_00')))
                        
                        segments.append({
                            'start_ts': float(start),
                            'end_ts': float(end),
                            'transcript': text,
                            'speaker': speaker
                        })
                        
                        if len(segments) == 1:
                            print(f"[Parser] First segment: start={start}, end={end}, speaker={speaker}")
                
                elif 'transcript' in data:
                    print("[Parser] Fallback: Using single transcript")
                    segments.append({
                        'start_ts': 0.0,
                        'end_ts': 0.0,
                        'transcript': data['transcript'],
                        'speaker': 'SPEAKER_00'
                    })
        
        print(f"\n[Parser] Total segments parsed: {len(segments)}")
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
                          (noun['syllables'] / count_syllables(text) * duration if count_syllables(text) > 0 else 0),
                'segment_idx': segments.index(seg),
                'speaker': seg.get('speaker', 'UNKNOWN')
            })
    
    return proper_nouns_timed

def translate_with_sarvam(
    text: str,
    sarvam_api_key: str,
    source_language: str = "hi-IN",
    target_language: str = "en-IN"
) -> str:
    client = SarvamAI(api_subscription_key=sarvam_api_key)
    
    sentences = re.split(r'([।|.!?]+)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunk_size = 10
    translated_chunks = []
    
    print(f"\n[Translation] Translating {len(sentences)} sentence fragments...")
    
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i + chunk_size])
        
        try:
            response = client.text.translate(
                input=chunk,
                source_language_code=source_language,
                target_language_code=target_language,
                mode="formal",
                model="sarvam-translate:v1",
                numerals_format="native",
                speaker_gender="Male",
                enable_preprocessing=True
            )
            
            translated = response.translated_text
            translated_chunks.append(translated)
            print(f"   ✓ Translated chunk {i//chunk_size + 1}/{(len(sentences)-1)//chunk_size + 1}")
        except Exception as e:
            print(f"   ✗ Translation error: {e}")
            translated_chunks.append(chunk) 
    
    return ' '.join(translated_chunks)

gemini_rate_limiter = RateLimiter(calls_per_minute=15)

def llm_trie_manager(full_text: str, context: str, gemini_api_key: str = None, language: str = "Hindi") -> str:
    if gemini_api_key is None:
        return full_text
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    words = full_text.split()
    chunk_size = 200
    overlap = 20
    corrected_chunks = []
    
    total_chunks = (len(words) + chunk_size - overlap - 1) // (chunk_size - overlap)
    print(f"  [LLM] Processing {total_chunks} chunks with rate limiting...")
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunk_num = i // (chunk_size - overlap) + 1
        
        if language == "Hindi":
            prompt = f"""Context: {context}

Original Hindi transcript (may have ASR errors):
{chunk}

Fix any obvious transcription errors while preserving the meaning. Only correct clear mistakes like:
- Wrong proper nouns (e.g., "इंडियन ऑइल कोऑपरेशन" → "इंडियन ऑयल कॉर्पोरेशन")
- Common ASR errors
- Spelling mistakes

Return ONLY the corrected Hindi text, nothing else:"""
        else:
            prompt = f"""Context: {context}

Original English translation (may have translation errors):
{chunk}

Fix any obvious translation or grammatical errors while preserving the meaning. Only correct clear mistakes like:
- Wrong proper nouns (e.g., "Indian Oil Cooperation" → "Indian Oil Corporation")
- Grammar issues
- Unnatural phrasing
- Technical term corrections

Return ONLY the corrected English text, nothing else:"""
        
        try:
            gemini_rate_limiter.wait_if_needed()
            
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
            print(f"   ✓ Processed chunk {chunk_num}/{total_chunks}")
        except Exception as e:
            print(f"   ✗ Gemini error on chunk {chunk_num}: {e}")
            corrected_chunks.append(chunk)
            time.sleep(2)
    
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
    
    print("\n[1/7] Transcribing with Sarvam Batch API (timestamps + diarization)...")
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
    
    print("\n[2/7] Calculating speaking rate...")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    total_syllables = count_syllables(full_transcript)
    spm = (total_syllables / duration) * 60 if duration > 0 else 0
    print(f"✓ Duration: {duration:.1f}s")
    print(f"✓ Syllables: {total_syllables}")
    print(f"✓ Speaking rate: {spm:.1f} syllables/minute")
    
    print("\n[3/7] Detecting proper nouns with timestamps...")
    proper_nouns_timed = find_proper_nouns_in_segments(segments)
    print(f"✓ Found {len(proper_nouns_timed)} proper nouns with timestamps:")
    for pn in proper_nouns_timed[:5]:
        print(f"   - {pn['text']} ({pn['start_ts']:.2f}s-{pn['end_ts']:.2f}s, {pn['speaker']})")
    if len(proper_nouns_timed) > 5:
        print(f"   ... and {len(proper_nouns_timed) - 5} more")
    
    print("\n[4/7] Re-translating proper noun segments...")
    corrected_transcript = full_transcript
    
    for idx, noun_info in enumerate(proper_nouns_timed[:10], 1):
        print(f"\n   [{idx}/{min(10, len(proper_nouns_timed))}] Correcting: {noun_info['text']}")
        print(f"         Time: {noun_info['start_ts']:.2f}s - {noun_info['end_ts']:.2f}s")
        print(f"         Speaker: {noun_info['speaker']}")
        
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
            print(f"         ✓ Replaced with: {better_translation}")
        except Exception as e:
            print(f"         ✗ Error: {e}")
            continue
    
    if use_llm and gemini_api_key:
        print("\n[5/7] Applying Gemini LLM corrections on Hindi (with rate limiting)...")
        try:
            corrected_transcript = llm_trie_manager(
                corrected_transcript,
                context,
                gemini_api_key,
                language="Hindi"
            )
            print("✓ Hindi corrections applied")
        except Exception as e:
            print(f"✗ Gemini correction failed: {e}")
    else:
        print("\n[5/7] Skipping Hindi LLM corrections")
    
    print("\n[6/7] Translating Hindi to English with Sarvam API...")
    try:
        english_translation = translate_with_sarvam(
            corrected_transcript,
            sarvam_api_key,
            source_language="hi-IN",
            target_language="en-IN"
        )
        print("✓ Translation completed")
        print(f"✓ English translation length: {len(english_translation)} chars")
        print(f"Preview: {english_translation[:200]}...")
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        english_translation = ""
    
    if use_llm and gemini_api_key and english_translation:
        print("\n[7/7] Applying Gemini LLM corrections on English (with rate limiting)...")
        try:
            corrected_english = llm_trie_manager(
                english_translation,
                context,
                gemini_api_key,
                language="English"
            )
            print("✓ English corrections applied")
        except Exception as e:
            print(f"✗ English correction failed: {e}")
            corrected_english = english_translation
    else:
        print("\n[7/7] Skipping English LLM corrections")
        corrected_english = english_translation
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n[Hindi] Original (first 500 chars):\n{full_transcript[:500]}...")
    print(f"\n[Hindi] Corrected (first 500 chars):\n{corrected_transcript[:500]}...")
    print(f"\n[English] Translation (first 500 chars):\n{english_translation[:500] if english_translation else 'N/A'}...")
    print(f"\n[English] Corrected (first 500 chars):\n{corrected_english[:500] if corrected_english else 'N/A'}...")
    
    chars_changed = sum(c1 != c2 for c1, c2 in zip(full_transcript, corrected_transcript))
    print(f"\nHindi changes made: {chars_changed} characters")
    print(f"Proper nouns processed: {min(10, len(proper_nouns_timed))}")
    
    return {
        'hindi_original': full_transcript,
        'hindi_corrected': corrected_transcript,
        'english_translation': english_translation,
        'english_corrected': corrected_english,
        'segments': segments
    }

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
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not SARVAM_API_KEY:
        print("Error: SARVAM_API_KEY not found in .env file.")
        exit(1)
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in .env file. LLM corrections will be skipped.")
    
    results = algorithm(
        audio_path=AUDIO_FILE,
        sarvam_api_key=SARVAM_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        language="hi-IN",
        context="Corporate podcast about business, stock market, and Indian companies",
        num_speakers=2,
        use_llm=True
    )
    
    print("\n" + "=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)
    
    hindi_original_file = "hindi_original_transcript.txt"
    with open(hindi_original_file, "w", encoding="utf-8") as f:
        f.write(results['hindi_original'])
    print(f"\n✓ Hindi original saved to: {hindi_original_file}")
    
    hindi_corrected_file = "hindi_corrected_transcript.txt"
    with open(hindi_corrected_file, "w", encoding="utf-8") as f:
        f.write(results['hindi_corrected'])
    print(f"✓ Hindi corrected saved to: {hindi_corrected_file}")
    
    if results['english_translation']:
        english_translation_file = "english_translation.txt"
        with open(english_translation_file, "w", encoding="utf-8") as f:
            f.write(results['english_translation'])
        print(f"✓ English translation saved to: {english_translation_file}")
    
    if results['english_corrected']:
        english_corrected_file = "english_corrected_final.txt"
        with open(english_corrected_file, "w", encoding="utf-8") as f:
            f.write(results['english_corrected'])
        print(f"✓ English corrected (FINAL) saved to: {english_corrected_file}")
    
    segments_file = "segments_with_timestamps.json"
    with open(segments_file, "w", encoding="utf-8") as f:
        json.dump(results['segments'], f, indent=2, ensure_ascii=False)
    print(f"✓ Segments saved to: {segments_file}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nOutput files created:")
    print("  1. hindi_original_transcript.txt - Raw Hindi ASR output")
    print("  2. hindi_corrected_transcript.txt - Corrected Hindi with proper nouns")
    print("  3. english_translation.txt - Raw English translation from Sarvam")
    print("  4. english_corrected_final.txt - Final corrected English (USE THIS)")
    print("  5. segments_with_timestamps.json - Timestamped segments")
    print("=" * 80)