import os
import sys
import json
import argparse
import pickle
import requests
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import Counter

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not available. Install with: pip install jiwer")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")


@dataclass
class TranscriptionSegment:
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: float = 1.0


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.session = requests.Session()
        
    def generate(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": 2048,
                    }
                }
                
                response = self.session.post(
                    f"{self.base_url}?key={self.api_key}",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        text = result['candidates'][0]['content']['parts'][0]['text']
                        return text.strip()
                else:
                    print(f"Gemini API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Error calling Gemini (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return ""
    
    def generate_json(self, prompt: str, temperature: float = 0.1) -> Dict:
        response = self.generate(prompt, temperature)
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            print(f"Response: {response}")
            return {}


class GeminiNumberNormalizer:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client
        
    def normalize_text(self, text: str) -> str:
        prompt = f"""You are a number normalization expert for Hindi-English mixed text.

Task: Convert ALL number words (Hindi, English, Hinglish) to digits in the text below.

Rules:
1. Convert Hindi number words: दस -> 10, पांच -> 5, सौ -> 100
2. Convert English number words: ten -> 10, five -> 5, hundred -> 100
3. Convert Hinglish/transliterated: das -> 10, paanch -> 5, sau -> 100
4. Keep existing digits unchanged
5. Preserve ALL other text, punctuation, and formatting EXACTLY
6. Only change number words to digits, nothing else

Text:
{text}

Return ONLY the normalized text, nothing else."""

        normalized = self.gemini.generate(prompt, temperature=0.0)
        return normalized if normalized else text
    
    def extract_numbers(self, text: str) -> List[str]:
        prompt = f"""Extract all numbers from the following text and return them as a JSON list.

Text:
{text}

Return format:
```json
{{"numbers": ["10", "5", "100.34"]}}
```

Extract numbers whether they are:
- Digits: 10, 100.34
- Hindi words: दस, पांच
- English words: ten, five
- Hinglish: das, paanch

Convert all to digit strings in the output."""

        result = self.gemini.generate_json(prompt)
        return result.get('numbers', [])


class GeminiProperNounExtractor:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client
        
    def extract_proper_nouns(self, text: str, context: str = "stock_market") -> Dict[str, List[str]]:
        context_info = ""
        if context == "stock_market":
            context_info = """
Context: This is a stock market/trading conversation in India.
Pay special attention to:
- Company names: HDFC, ICICI, TCS, Infosys, Reliance, etc.
- Exchanges: BSE, NSE, Bombay Stock Exchange, National Stock Exchange
- Indices: Nifty, Sensex, Bank Nifty
- Financial entities: SEBI, RBI
- Stock codes: 2-6 letter uppercase codes
"""
        
        prompt = f"""{context_info}

Extract ALL proper nouns from the text below and categorize them.

Text:
{text}

Return a JSON with these categories:
- PERSON: People names
- ORG: Organizations, companies
- LOCATION: Places, exchanges
- PRODUCT: Financial products, indices
- TICKER: Stock codes/tickers

Format:
```json
{{
  "PERSON": ["name1", "name2"],
  "ORG": ["HDFC", "TCS"],
  "LOCATION": ["NSE", "BSE"],
  "PRODUCT": ["Nifty", "Sensex"],
  "TICKER": ["RELIANCE", "INFY"]
}}
```

Important:
- Normalize to lowercase for consistency
- Handle abbreviations: HDFC = HDFC Bank
- Include both English and Hindi proper nouns
- Be comprehensive - extract ALL proper nouns"""

        result = self.gemini.generate_json(prompt)
        
        categories = ['PERSON', 'ORG', 'LOCATION', 'PRODUCT', 'TICKER']
        for cat in categories:
            if cat not in result:
                result[cat] = []
        
        return result
    
    def match_proper_nouns(self, entity1: str, entity2: str, context: str = "stock_market") -> bool:
        prompt = f"""Do these two entity names refer to the SAME entity in the stock market context?

Entity 1: {entity1}
Entity 2: {entity2}

Context: Indian stock market (companies, exchanges, financial entities)

Consider:
- Abbreviations: HDFC = HDFC Bank
- Full names: BSE = Bombay Stock Exchange
- Common variations: TCS = Tata Consultancy Services
- Typos/misspellings: HSBC vs HDFC (these are DIFFERENT)

Return ONLY "YES" if they are the same entity, or "NO" if different.
One word answer only."""

        response = self.gemini.generate(prompt, temperature=0.0).upper()
        return "YES" in response
    
    def calculate_proper_noun_score(self, ref_entities: Dict, hyp_entities: Dict) -> Tuple[float, Dict]:
        total_entities = 0
        matched_entities = 0
        details = {}
        
        for category in ref_entities.keys():
            ref_list = ref_entities.get(category, [])
            hyp_list = hyp_entities.get(category, [])
            
            if not ref_list:
                details[category] = {
                    'reference': [],
                    'hypothesis': hyp_list,
                    'matched': [],
                    'missing': [],
                    'score': 1.0
                }
                continue
            
            matched = []
            missing = []
            
            for ref_entity in ref_list:
                total_entities += 1
                found = False
                
                if ref_entity.lower() in [h.lower() for h in hyp_list]:
                    found = True
                    matched.append(ref_entity)
                else:
                    for hyp_entity in hyp_list:
                        if self.match_proper_nouns(ref_entity, hyp_entity):
                            found = True
                            matched.append(ref_entity)
                            break
                
                if found:
                    matched_entities += 1
                else:
                    missing.append(ref_entity)
            
            category_score = len(matched) / len(ref_list) if ref_list else 1.0
            
            details[category] = {
                'reference': ref_list,
                'hypothesis': hyp_list,
                'matched': matched,
                'missing': missing,
                'score': category_score
            }
        
        overall_score = matched_entities / total_entities if total_entities > 0 else 1.0
        
        return overall_score, details


class SarvamAITranscriber:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sarvam.ai/speech-to-text"
        
    def transcribe(self, audio_path: str, language: str = "hi-IN") -> Optional[Dict]:
        print(f"Transcribing {audio_path} with Sarvam AI...")
        
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'file': audio_file}
                headers = {'API-Key': self.api_key}
                data = {
                    'language': language, 
                    'with_timestamps': 'true',
                    'enable_diarization': 'true'
                }
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._parse_response(result)
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def _parse_response(self, response: Dict) -> Dict:
        segments = []
        full_transcript = []
        
        if 'segments' in response:
            for seg in response['segments']:
                segment = TranscriptionSegment(
                    text=seg.get('text', ''),
                    start_time=seg.get('start', 0.0),
                    end_time=seg.get('end', 0.0),
                    speaker=seg.get('speaker', None),
                    confidence=seg.get('confidence', 1.0)
                )
                segments.append(segment)
                
                speaker_label = segment.speaker if segment.speaker else ''
                full_transcript.append(f"{speaker_label}: {segment.text}" if speaker_label else segment.text)
        
        return {
            'transcript': '\n'.join(full_transcript),
            'segments': segments,
            'language': response.get('language', 'hi-IN')
        }


class MalformedTextDetector:
    @staticmethod
    def detect_repetitions(text: str) -> float:
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        repetition_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetition_count += 1
        
        repetition_ratio = repetition_count / len(words)
        return repetition_ratio * 5.0


class SemanticSimilarityScorer:
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini = gemini_client
        self.model = None
        
        if not gemini_client and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                pass
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        if self.gemini:
            return self._gemini_similarity(text1, text2)
        elif self.model:
            return self._transformer_similarity(text1, text2)
        else:
            return self._simple_similarity(text1, text2)
    
    def _gemini_similarity(self, text1: str, text2: str) -> float:
        prompt = f"""Rate the semantic similarity between these two texts on a scale of 0.0 to 1.0.

Text 1: {text1}

Text 2: {text2}

Consider:
- Overall meaning and intent
- Key information preservation
- Hindi-English code-switching equivalence (das = 10, दस = ten)
- Synonym usage

Return ONLY a number between 0.0 and 1.0, nothing else.
Example: 0.85"""

        try:
            response = self.gemini.generate(prompt, temperature=0.0)
            score = float(re.search(r'\d+\.?\d*', response).group())
            return np.clip(score, 0.0, 1.0)
        except:
            return self._simple_similarity(text1, text2)
    
    def _transformer_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, similarity)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0


class SpeakerDiarizationScorer:
    @staticmethod
    def parse_speaker_segments(text: str) -> List[Tuple[str, str]]:
        lines = text.strip().split('\n')
        segments = []
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip()
                segments.append((speaker, text))
            else:
                if segments:
                    segments[-1] = (segments[-1][0], segments[-1][1] + ' ' + line)
        
        return segments
    
    @staticmethod
    def calculate_speaker_accuracy(ref_segments: List[Tuple[str, str]],
                                   hyp_segments: List[Tuple[str, str]]) -> float:
        if not ref_segments or not hyp_segments:
            return 0.0
        
        correct = 0
        total = min(len(ref_segments), len(hyp_segments))
        
        for i in range(total):
            if ref_segments[i][0].lower() == hyp_segments[i][0].lower():
                correct += 1
        
        return correct / total if total > 0 else 0.0


class WERCalculator:
    def __init__(self, number_normalizer: GeminiNumberNormalizer):
        self.normalizer = number_normalizer
    
    def preprocess_text(self, text: str) -> str:
        text = self.normalizer.normalize_text(text)
        text = re.sub(r'^\w+:\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\(\d+:\d+\.?\d*-\d+:\d+\.?\d*\)', '', text)
        text = ' '.join(text.split())
        return text.lower().strip()
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_clean = self.preprocess_text(reference)
        hyp_clean = self.preprocess_text(hypothesis)
        
        if JIWER_AVAILABLE:
            return jiwer.wer(ref_clean, hyp_clean)
        else:
            return self._simple_wer(ref_clean, hyp_clean)
    
    @staticmethod
    def _simple_wer(reference: str, hypothesis: str) -> float:
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,
                        d[i][j-1] + 1,
                        d[i-1][j-1] + 1
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class RewardModelTrainer:
    def __init__(self, gemini_api_key: str):
        self.gemini = GeminiClient(gemini_api_key)
        self.number_normalizer = GeminiNumberNormalizer(self.gemini)
        self.proper_noun_extractor = GeminiProperNounExtractor(self.gemini)
        self.wer_calculator = WERCalculator(self.number_normalizer)
        self.malformed_detector = MalformedTextDetector()
        self.semantic_scorer = SemanticSimilarityScorer(self.gemini)
        self.speaker_scorer = SpeakerDiarizationScorer()
        
    def extract_features(self, reference_text: str, hypothesis_text: str) -> Dict[str, float]:
        features = {}
        print("\nFeature Extraction (Gemini-Enhanced):")
        
        wer = self.wer_calculator.calculate_wer(reference_text, hypothesis_text)
        features['wer'] = wer
        print(f"  WER (Gemini-normalized): {wer:.4f}")
        
        ref_entities = self.proper_noun_extractor.extract_proper_nouns(reference_text)
        hyp_entities = self.proper_noun_extractor.extract_proper_nouns(hypothesis_text)
        
        pn_score, pn_details = self.proper_noun_extractor.calculate_proper_noun_score(
            ref_entities, hyp_entities
        )
        features['proper_noun_score'] = pn_score
        features['proper_noun_penalty'] = (1.0 - pn_score) * 10.0
        features['proper_noun_details'] = pn_details
        
        print(f"  Proper Noun Score: {pn_score:.4f}")
        print(f"  Proper Noun Penalty (10x): {features['proper_noun_penalty']:.4f}")
        
        for category, info in pn_details.items():
            if info['missing']:
                print(f"    Missing {category}: {info['missing']}")
        
        repetition_penalty = self.malformed_detector.detect_repetitions(hypothesis_text)
        features['repetition_penalty'] = repetition_penalty
        print(f"  Repetition Penalty: {repetition_penalty:.4f}")
        
        semantic_sim = self.semantic_scorer.calculate_similarity(reference_text, hypothesis_text)
        features['semantic_similarity'] = semantic_sim
        print(f"  Semantic Similarity (Gemini): {semantic_sim:.4f}")
        
        ref_segments = self.speaker_scorer.parse_speaker_segments(reference_text)
        hyp_segments = self.speaker_scorer.parse_speaker_segments(hypothesis_text)
        speaker_acc = self.speaker_scorer.calculate_speaker_accuracy(ref_segments, hyp_segments)
        features['speaker_accuracy'] = speaker_acc
        print(f"  Speaker Accuracy: {speaker_acc:.4f}")
        
        ref_numbers = set(self.number_normalizer.extract_numbers(reference_text))
        hyp_numbers = set(self.number_normalizer.extract_numbers(hypothesis_text))
        
        if ref_numbers:
            number_accuracy = len(ref_numbers & hyp_numbers) / len(ref_numbers)
        else:
            number_accuracy = 1.0
        features['number_accuracy'] = number_accuracy
        print(f"  Number Accuracy (Gemini): {number_accuracy:.4f}")
        
        return features
    
    def calculate_reward_score(self, features: Dict[str, float]) -> float:
        wer_score = max(0, 1.0 - features['wer'])
        proper_noun_score = features['proper_noun_score']
        repetition_score = max(0, 1.0 - features['repetition_penalty'] / 5.0)
        semantic_score = features['semantic_similarity']
        speaker_score = features['speaker_accuracy']
        number_score = features['number_accuracy']
        
        weights = {
            'wer': 0.15,
            'proper_noun': 0.40,
            'repetition': 0.10,
            'semantic': 0.15,
            'speaker': 0.10,
            'number': 0.10
        }
        
        final_score = (
            weights['wer'] * wer_score +
            weights['proper_noun'] * proper_noun_score +
            weights['repetition'] * repetition_score +
            weights['semantic'] * semantic_score +
            weights['speaker'] * speaker_score +
            weights['number'] * number_score
        )
        
        return np.clip(final_score, 0.0, 1.0)
    
    def save_model(self, filepath: str):
        model_data = {
            'weights': {
                'wer': 0.15,
                'proper_noun': 0.40,
                'repetition': 0.10,
                'semantic': 0.15,
                'speaker': 0.10,
                'number': 0.10
            },
            'version': '4.0-gemini-integration'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")


def load_ground_truth_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} ground truth samples")
    return data


def main():
    parser = argparse.ArgumentParser(description="ASR Reward Model with Gemini 2.0 Flash")
    parser.add_argument("--mode", choices=["evaluate", "test"], 
                       default="test", help="Operation mode")
    parser.add_argument("--gemini_api_key", required=True,
                       help="Gemini API key")
    parser.add_argument("--sarvam_api_key",
                       help="Sarvam AI API key (for evaluate mode)")
    parser.add_argument("--ground_truth", default="ground_truth.json",
                       help="Path to ground truth JSON file")
    parser.add_argument("--model", default="reward_model_gemini.pkl", 
                       help="Model file path")
    parser.add_argument("--output", default="evaluation_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    trainer = RewardModelTrainer(args.gemini_api_key)
    
    if args.mode == "test":
        print("=" * 70)
        print("TEST MODE: Stock Market Conversation with Gemini 2.0 Flash")
        print("=" * 70)
        reference = "buy HDFC"
        hypothesis = "buy HSBC"
        
        print("\nREFERENCE (Ground Truth):")
        print(reference)
        print("\nHYPOTHESIS (ASR Output):")
        print(hypothesis)
        
        print("\nNote: Gemini will:")
        print("  - Normalize 'das' to '10' intelligently")
        print("  - Detect HDFC vs HSBC mismatch (10x penalty)")
        print("  - Calculate semantic similarity")
        
        features = trainer.extract_features(reference, hypothesis)
        score = trainer.calculate_reward_score(features)
        
        print("\n" + "=" * 70)
        print(f"FINAL REWARD SCORE: {score:.4f}")
        print("=" * 70)
        
        print("\nScore Breakdown:")
        print(f"  WER Component: {max(0, 1.0 - features['wer']) * 0.15:.4f} (15%)")
        print(f"  Proper Noun Component: {features['proper_noun_score'] * 0.40:.4f} (40%, Gemini-matched)")
        print(f"  Repetition Component: {max(0, 1.0 - features['repetition_penalty']/5.0) * 0.10:.4f} (10%)")
        print(f"  Semantic Component: {features['semantic_similarity'] * 0.15:.4f} (15%, Gemini)")
        print(f"  Speaker Component: {features['speaker_accuracy'] * 0.10:.4f} (10%)")
        print(f"  Number Component: {features['number_accuracy'] * 0.10:.4f} (10%, Gemini)")
        
        trainer.save_model(args.model)
        
    elif args.mode == "evaluate":
        if not args.sarvam_api_key:
            print("Error: --sarvam_api_key required for evaluate mode")
            sys.exit(1)
        
        transcriber = SarvamAITranscriber(args.sarvam_api_key)
        ground_truth = load_ground_truth_data(args.ground_truth)
        
        results = []
        for idx, item in enumerate(ground_truth, 1):
            audio_path = item['audio_path']
            reference_text = item['reference_text']
            
            print(f"\n{'='*70}")
            print(f"Processing [{idx}/{len(ground_truth)}]: {audio_path}")
            print(f"{'='*70}")
            
            transcription = transcriber.transcribe(audio_path)
            
            if transcription:
                hypothesis_text = transcription['transcript']
                
                print("\nReference:")
                print(reference_text)
                print("\nHypothesis:")
                print(hypothesis_text)
                
                features = trainer.extract_features(reference_text, hypothesis_text)
                score = trainer.calculate_reward_score(features)
                
                result = {
                    'audio_path': audio_path,
                    'reference': reference_text,
                    'hypothesis': hypothesis_text,
                    'features': {
                        'wer': float(features['wer']),
                        'proper_noun_score': float(features['proper_noun_score']),
                        'proper_noun_penalty': float(features['proper_noun_penalty']),
                        'repetition_penalty': float(features['repetition_penalty']),
                        'semantic_similarity': float(features['semantic_similarity']),
                        'speaker_accuracy': float(features['speaker_accuracy']),
                        'number_accuracy': float(features['number_accuracy'])
                    },
                    'proper_noun_details': features['proper_noun_details'],
                    'reward_score': float(score)
                }
                
                results.append(result)
                print(f"\n✓ Reward Score: {score:.4f}")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"Results saved to {args.output}")
        
        if results:
            scores = [r['reward_score'] for r in results]
            print(f"\n{'='*70}")
            print("SUMMARY STATISTICS")
            print(f"{'='*70}")
            print(f"  Total Samples: {len(results)}")
            print(f"  Average Score: {np.mean(scores):.4f}")
            print(f"  Std Dev: {np.std(scores):.4f}")
            print(f"  Min Score: {np.min(scores):.4f}")
            print(f"  Max Score: {np.max(scores):.4f}")
            print(f"  Median Score: {np.median(scores):.4f}")
            
            excellent = sum(1 for s in scores if s >= 0.8)
            good = sum(1 for s in scores if 0.6 <= s < 0.8)
            fair = sum(1 for s in scores if 0.4 <= s < 0.6)
            poor = sum(1 for s in scores if s < 0.4)
            
            print(f"\nScore Distribution:")
            print(f"  Excellent (≥0.8): {excellent:3d} ({excellent/len(results)*100:5.1f}%)")
            print(f"  Good (0.6-0.8):  {good:3d} ({good/len(results)*100:5.1f}%)")
            print(f"  Fair (0.4-0.6):  {fair:3d} ({fair/len(results)*100:5.1f}%)")
            print(f"  Poor (<0.4):     {poor:3d} ({poor/len(results)*100:5.1f}%)")
            
            avg_wer = np.mean([r['features']['wer'] for r in results])
            avg_pn_score = np.mean([r['features']['proper_noun_score'] for r in results])
            avg_semantic = np.mean([r['features']['semantic_similarity'] for r in results])
            avg_speaker = np.mean([r['features']['speaker_accuracy'] for r in results])
            avg_number = np.mean([r['features']['number_accuracy'] for r in results])
            
            print(f"\nAverage Metrics:")
            print(f"  WER: {avg_wer:.4f}")
            print(f"  Proper Noun Score: {avg_pn_score:.4f}")
            print(f"  Semantic Similarity: {avg_semantic:.4f}")
            print(f"  Speaker Accuracy: {avg_speaker:.4f}")
            print(f"  Number Accuracy: {avg_number:.4f}")
            
            total_proper_nouns = 0
            total_missing = 0
            category_stats = {}
            
            for r in results:
                for category, info in r['proper_noun_details'].items():
                    if category not in category_stats:
                        category_stats[category] = {'total': 0, 'missing': 0}
                    
                    category_stats[category]['total'] += len(info['reference'])
                    category_stats[category]['missing'] += len(info['missing'])
                    
                    total_proper_nouns += len(info['reference'])
                    total_missing += len(info['missing'])
            
            if total_proper_nouns > 0:
                print(f"\n{'='*70}")
                print("PROPER NOUN ANALYSIS (Gemini-Matched)")
                print(f"{'='*70}")
                print(f"  Total Proper Nouns: {total_proper_nouns}")
                print(f"  Total Missing: {total_missing}")
                print(f"  Overall Accuracy: {(1 - total_missing/total_proper_nouns)*100:.1f}%")
                
                print(f"\nBy Category:")
                for category, stats in category_stats.items():
                    if stats['total'] > 0:
                        accuracy = (1 - stats['missing']/stats['total']) * 100
                        print(f"  {category:12s}: {stats['total']:3d} total, "
                              f"{stats['missing']:3d} missing, "
                              f"{accuracy:5.1f}% accuracy")
        
        trainer.save_model(args.model)
        print(f"\n{'='*70}")


if __name__ == "__main__":
    main()