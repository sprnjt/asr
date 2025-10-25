import os
import sys
import json
import argparse
import pickle
import requests
import re
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from collections import Counter

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa/soundfile not available. Install with: pip install librosa soundfile")

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


@dataclass
class AugmentationConfig:
    add_white_noise: bool = True
    noise_level: float = 0.005
    add_background_noise: bool = False
    background_noise_path: Optional[str] = None
    time_stretch: bool = True
    stretch_rate: float = 0.95
    pitch_shift: bool = True
    pitch_steps: int = 1
    add_reverb: bool = False
    volume_adjustment: bool = True
    volume_factor: float = 0.9


@dataclass
class AUHarnessMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    ece: float
    brier_score: float
    nll: float
    auroc: float
    auprc: float


class AudioAugmenter:
    def __init__(self, config: AugmentationConfig):
        if not AUDIO_AVAILABLE:
            raise ImportError("librosa and soundfile are required for audio augmentation")
        
        self.config = config
        self.background_noise = None
        
        if config.add_background_noise and config.background_noise_path:
            try:
                self.background_noise, _ = librosa.load(config.background_noise_path, sr=None)
                print(f"Loaded background noise from {config.background_noise_path}")
            except Exception as e:
                print(f"Warning: Could not load background noise: {e}")
    
    def add_white_noise(self, audio: np.ndarray, noise_level: float) -> np.ndarray:
        noise = np.random.randn(len(audio))
        augmented = audio + noise_level * noise
        return augmented
    
    def add_background_noise(self, audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
        if self.background_noise is None:
            return audio
        
        if len(self.background_noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(self.background_noise)))
            background = np.tile(self.background_noise, repeats)[:len(audio)]
        else:
            start_idx = random.randint(0, len(self.background_noise) - len(audio))
            background = self.background_noise[start_idx:start_idx + len(audio)]
        
        audio_power = np.mean(audio ** 2)
        background_power = np.mean(background ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        scale_factor = np.sqrt(audio_power / (background_power * snr_linear))
        
        augmented = audio + scale_factor * background
        return augmented
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, sr: int, steps: int) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    
    def add_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        impulse_response = np.zeros(int(0.1 * sr))
        impulse_response[0] = 1
        for i in range(1, len(impulse_response)):
            impulse_response[i] = impulse_response[i-1] * 0.5 * np.random.random()
        
        augmented = np.convolve(audio, impulse_response, mode='same')
        return augmented / np.max(np.abs(augmented))
    
    def adjust_volume(self, audio: np.ndarray, factor: float) -> np.ndarray:
        return audio * factor
    
    def augment(self, audio_path: str, output_path: str, augmentation_type: str = "all") -> str:
        audio, sr = librosa.load(audio_path, sr=None)
        augmented = audio.copy()
        
        if augmentation_type == "all" or augmentation_type == "white_noise":
            if self.config.add_white_noise:
                augmented = self.add_white_noise(augmented, self.config.noise_level)
        
        if augmentation_type == "all" or augmentation_type == "background_noise":
            if self.config.add_background_noise and self.background_noise is not None:
                augmented = self.add_background_noise(augmented)
        
        if augmentation_type == "all" or augmentation_type == "time_stretch":
            if self.config.time_stretch:
                augmented = self.time_stretch(augmented, self.config.stretch_rate)
        
        if augmentation_type == "all" or augmentation_type == "pitch_shift":
            if self.config.pitch_shift:
                augmented = self.pitch_shift(augmented, sr, self.config.pitch_steps)
        
        if augmentation_type == "all" or augmentation_type == "reverb":
            if self.config.add_reverb:
                augmented = self.add_reverb(augmented, sr)
        
        if augmentation_type == "all" or augmentation_type == "volume":
            if self.config.volume_adjustment:
                augmented = self.adjust_volume(augmented, self.config.volume_factor)
        
        augmented = np.clip(augmented, -1.0, 1.0)
        
        sf.write(output_path, augmented, sr)
        print(f"Augmented audio saved to: {output_path}")
        
        return output_path
    
    def generate_multiple_augmentations(self, audio_path: str, output_dir: str, 
                                       num_variants: int = 5) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(audio_path).stem
        augmentation_types = ["white_noise", "background_noise", "time_stretch", 
                             "pitch_shift", "reverb", "volume", "all"]
        
        augmented_files = []
        
        for i in range(num_variants):
            aug_type = random.choice(augmentation_types)
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}_{aug_type}.wav")
            
            try:
                augmented_path = self.augment(audio_path, output_path, aug_type)
                augmented_files.append(augmented_path)
            except Exception as e:
                print(f"Error augmenting {audio_path} with {aug_type}: {e}")
        
        return augmented_files


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
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


class AUHarnessEvaluator:
    def __init__(self, au_harness_path: Optional[str] = None):
        self.au_harness_path = au_harness_path or self._find_au_harness()
        
    def _find_au_harness(self) -> Optional[str]:
        possible_paths = [
            "./AU-Harness",
            "../AU-Harness",
            "~/AU-Harness",
            os.path.expanduser("~/AU-Harness")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _check_au_harness_installed(self) -> bool:
        if not self.au_harness_path:
            return False
        
        return os.path.exists(os.path.join(self.au_harness_path, "au_harness"))
    
    def install_au_harness(self):
        print("\n" + "="*70)
        print("INSTALLING AU-HARNESS")
        print("="*70)
        
        if self._check_au_harness_installed():
            print("AU-Harness already installed!")
            return True
        
        try:
            print("Cloning AU-Harness repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/ServiceNow/AU-Harness.git"
            ], check=True)
            
            self.au_harness_path = "./AU-Harness"
            
            print("Installing AU-Harness dependencies...")
            subprocess.run([
                "pip", "install", "-e", self.au_harness_path
            ], check=True)
            
            print("✓ AU-Harness installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing AU-Harness: {e}")
            return False
    
    def prepare_au_harness_data(self, results: List[Dict], output_path: str):
        au_data = {
            "predictions": [],
            "labels": [],
            "confidences": []
        }
        
        for result in results:
            reward_score = result['reward_score']
            
            prediction = 1 if reward_score >= 0.6 else 0
            label = 1 if reward_score >= 0.8 else 0
            confidence = reward_score
            
            au_data["predictions"].append(prediction)
            au_data["labels"].append(label)
            au_data["confidences"].append(confidence)
        
        with open(output_path, 'w') as f:
            json.dump(au_data, f, indent=2)
        
        print(f"AU-Harness data prepared: {output_path}")
        return output_path
    
    def run_au_harness_evaluation(self, data_path: str, output_dir: str) -> Dict:
        if not self._check_au_harness_installed():
            print("Error: AU-Harness not installed. Run with --install_au_harness first.")
            return {}
        
        print("\n" + "="*70)
        print("RUNNING AU-HARNESS EVALUATION")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            import sys
            sys.path.insert(0, self.au_harness_path)
            from au_harness import AUHarness
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            predictions = np.array(data['predictions'])
            labels = np.array(data['labels'])
            confidences = np.array(data['confidences'])
            
            harness = AUHarness()
            
            results = harness.evaluate(
                predictions=predictions,
                labels=labels,
                confidences=confidences
            )
            
            metrics = AUHarnessMetrics(
                accuracy=float(results.get('accuracy', 0.0)),
                precision=float(results.get('precision', 0.0)),
                recall=float(results.get('recall', 0.0)),
                f1_score=float(results.get('f1_score', 0.0)),
                ece=float(results.get('ece', 0.0)),
                brier_score=float(results.get('brier_score', 0.0)),
                nll=float(results.get('nll', 0.0)),
                auroc=float(results.get('auroc', 0.0)),
                auprc=float(results.get('auprc', 0.0))
            )
            
            metrics_path = os.path.join(output_dir, "au_harness_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
            
            print("\n" + "="*70)
            print("AU-HARNESS METRICS")
            print("="*70)
            print(f"  Accuracy:     {metrics.accuracy:.4f}")
            print(f"  Precision:    {metrics.precision:.4f}")
            print(f"  Recall:       {metrics.recall:.4f}")
            print(f"  F1-Score:     {metrics.f1_score:.4f}")
            print(f"  ECE:          {metrics.ece:.4f}")
            print(f"  Brier Score:  {metrics.brier_score:.4f}")
            print(f"  NLL:          {metrics.nll:.4f}")
            print(f"  AUROC:        {metrics.auroc:.4f}")
            print(f"  AUPRC:        {metrics.auprc:.4f}")
            print("="*70)
            
            return asdict(metrics)
            
        except ImportError as e:
            print(f"Error importing AU-Harness: {e}")
            print("Try installing with: pip install git+https://github.com/ServiceNow/AU-Harness.git")
            return {}
        except Exception as e:
            print(f"Error running AU-Harness evaluation: {e}")
            return {}


class RewardModelTrainer:
    def __init__(self, gemini_api_key: str, augmenter: Optional[AudioAugmenter] = None):
        self.gemini = GeminiClient(gemini_api_key)
        self.number_normalizer = GeminiNumberNormalizer(self.gemini)
        self.proper_noun_extractor = GeminiProperNounExtractor(self.gemini)
        self.wer_calculator = WERCalculator(self.number_normalizer)
        self.malformed_detector = MalformedTextDetector()
        self.semantic_scorer = SemanticSimilarityScorer(self.gemini)
        self.speaker_scorer = SpeakerDiarizationScorer()
        self.augmenter = augmenter
        
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
            'version': '5.0-augmentation-au-harness'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")


def load_ground_truth_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} ground truth samples")
    return data


def augment_dataset(ground_truth: List[Dict], augmenter: AudioAugmenter, 
                   augmented_dir: str, num_variants: int = 3) -> List[Dict]:
    os.makedirs(augmented_dir, exist_ok=True)
    
    augmented_dataset = []
    
    for idx, item in enumerate(ground_truth, 1):
        audio_path = item['audio_path']
        reference_text = item['reference_text']
        
        print(f"\n{'='*70}")
        print(f"Augmenting [{idx}/{len(ground_truth)}]: {audio_path}")
        print(f"{'='*70}")
        
        augmented_files = augmenter.generate_multiple_augmentations(
            audio_path, augmented_dir, num_variants
        )
        
        for aug_file in augmented_files:
            augmented_dataset.append({
                'audio_path': aug_file,
                'reference_text': reference_text,
                'original_audio': audio_path,
                'is_augmented': True
            })
    
    print(f"\n{'='*70}")
    print(f"Generated {len(augmented_dataset)} augmented samples")
    print(f"{'='*70}")
    
    return augmented_dataset


def main():
    parser = argparse.ArgumentParser(description="ASR Reward Model with AU-Harness")
    parser.add_argument("--mode", choices=["evaluate", "test", "augment", "install_au_harness"], 
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
    parser.add_argument("--augment_output", default="augmented_audio",
                       help="Directory for augmented audio files")
    parser.add_argument("--num_augments", type=int, default=3,
                       help="Number of augmented variants per audio")
    parser.add_argument("--noise_level", type=float, default=0.005,
                       help="White noise level (0.0-0.1)")
    parser.add_argument("--background_noise",
                       help="Path to background noise audio file")
    parser.add_argument("--augmented_dataset", default="augmented_dataset.json",
                       help="Output path for augmented dataset JSON")
    parser.add_argument("--run_au_harness", action="store_true",
                       help="Run AU-Harness evaluation after main evaluation")
    parser.add_argument("--au_harness_path",
                       help="Path to AU-Harness installation")
    parser.add_argument("--au_harness_output", default="au_harness_results",
                       help="Output directory for AU-Harness results")
    
    args = parser.parse_args()
    
    if args.mode == "install_au_harness":
        evaluator = AUHarnessEvaluator(args.au_harness_path)
        evaluator.install_au_harness()
        return
    
    augmenter = None
    if AUDIO_AVAILABLE:
        aug_config = AugmentationConfig(
            add_white_noise=True,
            noise_level=args.noise_level,
            add_background_noise=args.background_noise is not None,
            background_noise_path=args.background_noise,
            time_stretch=True,
            stretch_rate=random.uniform(0.9, 1.1),
            pitch_shift=True,
            pitch_steps=random.randint(-2, 2),
            add_reverb=True,
            volume_adjustment=True,
            volume_factor=random.uniform(0.8, 1.2)
        )
        augmenter = AudioAugmenter(aug_config)
    
    trainer = RewardModelTrainer(args.gemini_api_key, augmenter)
    
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
    
    elif args.mode == "augment":
        if not AUDIO_AVAILABLE:
            print("Error: librosa and soundfile required for augmentation mode")
            print("Install with: pip install librosa soundfile")
            sys.exit(1)
        
        print("=" * 70)
        print("AUGMENTATION MODE: Generating Data Augmentations")
        print("=" * 70)
        
        ground_truth = load_ground_truth_data(args.ground_truth)
        
        augmented_dataset = augment_dataset(
            ground_truth, 
            augmenter, 
            args.augment_output, 
            args.num_augments
        )
        
        combined_dataset = ground_truth + augmented_dataset
        
        with open(args.augmented_dataset, 'w', encoding='utf-8') as f:
            json.dump(combined_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"Augmented dataset saved to: {args.augmented_dataset}")
        print(f"  Original samples: {len(ground_truth)}")
        print(f"  Augmented samples: {len(augmented_dataset)}")
        print(f"  Total samples: {len(combined_dataset)}")
        print(f"{'='*70}")
        
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
            is_augmented = item.get('is_augmented', False)
            
            print(f"\n{'='*70}")
            print(f"Processing [{idx}/{len(ground_truth)}]: {audio_path}")
            if is_augmented:
                print(f"  [AUGMENTED from: {item.get('original_audio', 'unknown')}]")
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
                    'is_augmented': is_augmented,
                    'original_audio': item.get('original_audio', audio_path),
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
            original_scores = [r['reward_score'] for r in results if not r['is_augmented']]
            augmented_scores = [r['reward_score'] for r in results if r['is_augmented']]
            
            print(f"\n{'='*70}")
            print("SUMMARY STATISTICS")
            print(f"{'='*70}")
            print(f"  Total Samples: {len(results)}")
            print(f"  Original Samples: {len(original_scores)}")
            print(f"  Augmented Samples: {len(augmented_scores)}")
            print(f"\n  Overall Average Score: {np.mean(scores):.4f}")
            print(f"  Overall Std Dev: {np.std(scores):.4f}")
            print(f"  Overall Min Score: {np.min(scores):.4f}")
            print(f"  Overall Max Score: {np.max(scores):.4f}")
            print(f"  Overall Median Score: {np.median(scores):.4f}")
            
            if original_scores:
                print(f"\n  Original Average Score: {np.mean(original_scores):.4f}")
                print(f"  Original Std Dev: {np.std(original_scores):.4f}")
            
            if augmented_scores:
                print(f"\n  Augmented Average Score: {np.mean(augmented_scores):.4f}")
                print(f"  Augmented Std Dev: {np.std(augmented_scores):.4f}")
                print(f"  Score Delta (Original - Augmented): {np.mean(original_scores) - np.mean(augmented_scores):.4f}")
            
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
            
            if augmented_scores:
                print(f"\n{'='*70}")
                print("AUGMENTATION IMPACT ANALYSIS")
                print(f"{'='*70}")
                
                wer_original = np.mean([r['features']['wer'] for r in results if not r['is_augmented']])
                wer_augmented = np.mean([r['features']['wer'] for r in results if r['is_augmented']])
                
                print(f"  WER Impact:")
                print(f"    Original: {wer_original:.4f}")
                print(f"    Augmented: {wer_augmented:.4f}")
                print(f"    Delta: {wer_augmented - wer_original:+.4f}")
                
                pn_original = np.mean([r['features']['proper_noun_score'] for r in results if not r['is_augmented']])
                pn_augmented = np.mean([r['features']['proper_noun_score'] for r in results if r['is_augmented']])
                
                print(f"\n  Proper Noun Score Impact:")
                print(f"    Original: {pn_original:.4f}")
                print(f"    Augmented: {pn_augmented:.4f}")
                print(f"    Delta: {pn_augmented - pn_original:+.4f}")
            
            if args.run_au_harness:
                evaluator = AUHarnessEvaluator(args.au_harness_path)
                
                au_data_path = os.path.join(args.au_harness_output, "au_harness_data.json")
                evaluator.prepare_au_harness_data(results, au_data_path)
                
                au_metrics = evaluator.run_au_harness_evaluation(au_data_path, args.au_harness_output)
                
                if au_metrics:
                    print("\n" + "="*70)
                    print("AU-HARNESS EVALUATION COMPLETE")
                    print("="*70)
                    print(f"Results saved to: {args.au_harness_output}")
        
        trainer.save_model(args.model)
        print(f"\n{'='*70}")


if __name__ == "__main__":
    main()