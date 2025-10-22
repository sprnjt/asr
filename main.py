import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
from pydub import AudioSegment
import librosa
import soundfile as sf
from jiwer import wer, cer
from sklearn.model_selection import train_test_split
import re
import tempfile
import shutil
from dotenv import load_dotenv

try:
    from sarvamai import SarvamAI
except ImportError:
    print("SarvamAI library not found. Please install it.")
    class SarvamAI:
        def __init__(self, api_subscription_key: str):
            print("Mock SarvamAI client initialized.")
            self.speech_to_text = self._MockSpeechToText()

        class _MockSpeechToText:
            def transcribe(self, file, language_code, model):
                print(f"Mock transcribe called for lang: {language_code}")
                class MockResponse:
                    transcript = "This is a mock transcript."
                    words = [{"word": "This", "start": 0.1, "end": 0.5}]
                return MockResponse()


class SarvamTranscriber:
    def __init__(self, api_key: str):
        self.client = SarvamAI(api_subscription_key=api_key)

    def transcribe_audio(self, audio_path: str, language: str = "hi-IN") -> Dict:
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.speech_to_text.transcribe(
                    file=audio_file,
                    language_code=language,
                    model="saarika:v2.5"
                )

            transcript_text = response.transcript if hasattr(response, 'transcript') else str(response)

            words = []
            if hasattr(response, 'words'):
                words = response.words

            return {
                'transcript': transcript_text,
                'words': words,
                'language': language,
                'raw_response': response
            }

        except Exception as e:
            raise Exception(f"Sarvam API Error: {str(e)}")


class TranscriptCleaner:
    @staticmethod
    def clean_transcript(text: str) -> str:
        text = text.lower()

        fillers = ['um', 'uh', 'er', 'ah', 'hmm', 'umm', 'uhh']
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!\-\']', '', text)
        text = re.sub(r'([\.!?,])\1+', r'\1', text)

        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)

        return text.strip()

    @staticmethod
    def extract_segments(transcript_text: str, words_per_segment: int = 50) -> List[Dict]:
        words = transcript_text.split()
        segments = []

        if len(words) == 0:
            return []

        for i in range(0, len(words), words_per_segment):
            segment_words = words[i:i + words_per_segment]
            segment_text = ' '.join(segment_words)

            segments.append({
                'text': segment_text,
                'start_word': i,
                'end_word': min(i + words_per_segment, len(words)),
                'segment_id': len(segments)
            })

        return segments


class NoiseInjector:
    def __init__(self, audio_path: str):
        try:
            self.audio, self.sr = librosa.load(audio_path, sr=16000)
        except FileNotFoundError:
            print(f"Warning: Audio file not found at {audio_path}. Using mock audio.")
            self.sr = 16000
            self.audio = np.random.randn(self.sr * 10)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}. Using mock audio.")
            self.sr = 16000
            self.audio = np.random.randn(self.sr * 10)


    def add_white_noise(self, snr_db: float = 20) -> np.ndarray:
        signal_power = np.mean(self.audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(self.audio))
        return self.audio + noise

    def add_background_noise(self, noise_type: str = 'cafe', snr_db: float = 15) -> np.ndarray:
        if noise_type == 'cafe':
            noise = np.random.normal(0, 0.01, len(self.audio))
            noise = librosa.effects.preemphasis(noise, coef=0.5)
        elif noise_type == 'street':
            noise = np.random.normal(0, 0.015, len(self.audio))
        else:
            noise = np.random.normal(0, 0.005, len(self.audio))

        signal_power = np.mean(self.audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise_mean_power = np.mean(noise ** 2)
        if noise_mean_power == 0:
            print("Warning: Generated noise has zero power. Skipping noise addition.")
            return self.audio
            
        noise = noise * np.sqrt(noise_power / noise_mean_power)

        return self.audio + noise

    def save_noisy_audio(self, noisy_audio: np.ndarray, output_path: str):
        sf.write(output_path, noisy_audio, self.sr)


class ASRQualityDataset:
    def __init__(self):
        self.data = []

    def create_synthetic_pairs(self, clean_transcript: str, num_variations: int = 5) -> List[Dict]:
        variations = []

        variations.append({
            'transcript': clean_transcript,
            'quality_score': 5,
            'wer': 0.0,
            'label': 'perfect'
        })

        words = clean_transcript.split()

        if len(words) == 0:
            return variations

        level_4 = clean_transcript.lower().replace('.', '').replace(',', '')
        variations.append({
            'transcript': level_4,
            'quality_score': 4,
            'wer': 0.0,
            'label': 'excellent'
        })

        level_3_words = words.copy()
        num_errors = max(1, len(words) // 15)
        for _ in range(num_errors):
            idx = np.random.randint(0, len(level_3_words))
            level_3_words[idx] = self._similar_word(level_3_words[idx])
        level_3 = ' '.join(level_3_words)
        variations.append({
            'transcript': level_3,
            'quality_score': 3,
            'wer': wer(clean_transcript, level_3),
            'label': 'good'
        })

        level_2_words = words.copy()
        num_errors = max(2, len(words) // 6)
        for _ in range(num_errors):
            if len(level_2_words) == 0: break
            idx = np.random.randint(0, len(level_2_words))
            if np.random.random() < 0.5:
                level_2_words[idx] = self._similar_word(level_2_words[idx])
            else:
                if len(level_2_words) > 1:
                    level_2_words.pop(idx)
        level_2 = ' '.join(level_2_words)
        variations.append({
            'transcript': level_2,
            'quality_score': 2,
            'wer': wer(clean_transcript, level_2),
            'label': 'fair'
        })

        level_1_words = words.copy()
        num_errors = max(3, len(words) // 3)
        for _ in range(num_errors):
            if len(level_1_words) == 0:
                break
            idx = np.random.randint(0, len(level_1_words))
            action = np.random.choice(['replace', 'delete', 'duplicate'])
            if action == 'replace':
                level_1_words[idx] = self._random_word()
            elif action == 'delete' and len(level_1_words) > 1:
                level_1_words.pop(idx)
            elif action == 'duplicate':
                level_1_words.insert(idx, level_1_words[idx])
        level_1 = ' '.join(level_1_words)
        variations.append({
            'transcript': level_1,
            'quality_score': 1,
            'wer': wer(clean_transcript, level_1),
            'label': 'poor'
        })

        return variations

    @staticmethod
    def _similar_word(word: str) -> str:
        if len(word) < 2:
            return word

        transformations = [
            lambda w: w[:-1] + ('s' if w[-1] != 's' else 'z'),
            lambda w: w.replace('ph', 'f'),
            lambda w: w.replace('c', 'k'),
            lambda w: w + 'e' if len(w) > 2 else w,
            lambda w: w[1:] if len(w) > 3 else w,
        ]

        transform = np.random.choice(transformations)
        return transform(word)

    @staticmethod
    def _random_word() -> str:
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on']
        return np.random.choice(common_words)


class ASRRewardDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item['transcript'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['quality_score'] - 1, dtype=torch.long)
        }

class RewardModelTrainer:
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            problem_type="single_label_classification"
        )

    def prepare_data(self, transcript_variations: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        min_samples_per_class = min(
            sum(1 for d in transcript_variations if d['quality_score'] == score)
            for score in range(1, 6)
        )

        use_stratify = len(transcript_variations) >= 50 and min_samples_per_class >= 2

        if use_stratify:
            train_data, val_data = train_test_split(
                transcript_variations,
                test_size=0.2,
                random_state=42,
                stratify=[d['quality_score'] for d in transcript_variations]
            )
        else:
            train_data, val_data = train_test_split(
                transcript_variations,
                test_size=0.2,
                random_state=42
            )

        if len(val_data) == 0 and len(train_data) > 0:
            val_data = [train_data[-1]]
            train_data = train_data[:-1]
        elif len(train_data) == 0 and len(val_data) > 0:
            train_data = [val_data[-1]]
            val_data = val_data[:-1]
        elif len(train_data) == 0 and len(val_data) == 0:
            print("Error: No data to prepare.")
            dummy_transcript = "dummy text"
            dummy_item = {
                'transcript': dummy_transcript,
                'quality_score': 1,
                'wer': 1.0,
                'label': 'poor'
            }
            train_data = [dummy_item]
            val_data = [dummy_item]


        train_dataset = ASRRewardDataset(train_data, self.tokenizer)
        val_dataset = ASRRewardDataset(val_data, self.tokenizer)

        train_batch_size = min(8, len(train_data)) if len(train_data) > 0 else 1
        val_batch_size = min(8, len(val_data)) if len(val_data) > 0 else 1

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs: int = 3, output_dir: str = "./reward_model"):
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory for training: {temp_dir}")
            
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f'{temp_dir}/logs',
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                save_safetensors=True,
                save_total_limit=2,
            )

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                accuracy = (predictions == labels).mean()
                return {"accuracy": accuracy}

            train_data = []
            for batch in train_loader:
                if batch['input_ids'].dim() == 0: continue
                for i in range(len(batch['input_ids'])):
                    train_data.append({
                        'input_ids': batch['input_ids'][i].tolist(),
                        'attention_mask': batch['attention_mask'][i].tolist(),
                        'labels': batch['labels'][i].item()
                    })

            val_data = []
            for batch in val_loader:
                if batch['input_ids'].dim() == 0: continue
                for i in range(len(batch['input_ids'])):
                    val_data.append({
                        'input_ids': batch['input_ids'][i].tolist(),
                        'attention_mask': batch['attention_mask'][i].tolist(),
                        'labels': batch['labels'][i].item()
                    })
            
            if not train_data:
                print("Error: No training data available for Trainer.")
                dummy_encoding = self.tokenizer("dummy", truncation=True, max_length=512, padding='max_length')
                train_data = [{
                    'input_ids': dummy_encoding['input_ids'],
                    'attention_mask': dummy_encoding['attention_mask'],
                    'labels': 0
                }]
            
            if not val_data:
                print("Warning: No validation data available for Trainer.")
                dummy_encoding = self.tokenizer("dummy", truncation=True, max_length=512, padding='max_length')
                val_data = [{
                    'input_ids': dummy_encoding['input_ids'],
                    'attention_mask': dummy_encoding['attention_mask'],
                    'labels': 0
                }]


            train_dataset = HFDataset.from_list(train_data)
            val_dataset = HFDataset.from_list(val_data)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            
            print(f"Copying trained model from {temp_dir} to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"✓ Model successfully saved to {output_dir}")
            
        return trainer

    def predict_quality(self, transcript: str) -> Dict:
        inputs = self.tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        quality_labels = ['poor', 'fair', 'good', 'excellent', 'perfect']

        return {
            'quality_score': predicted_class + 1,
            'quality_label': quality_labels[predicted_class],
            'confidence': confidence,
            'all_probabilities': probs[0].tolist()
        }


def main_pipeline(
    audio_path: str,
    sarvam_api_key: str,
    output_dir: str = "./asr_pipeline_output",
    language: str = "hi-IN"
):
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ASR QUALITY REWARD MODEL PIPELINE")
    print("=" * 80)

    print(f"\n[1/5] Transcribing audio with Sarvam AI (language: {language})...")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        print("Creating dummy audio file 'dummy_audio.wav' to proceed...")
        sr = 16000
        dummy_audio = np.random.randn(sr * 10)
        sf.write("dummy_audio.wav", dummy_audio, sr)
        audio_path = "dummy_audio.wav"
        
    transcriber = SarvamTranscriber(sarvam_api_key)
    try:
        transcript_data = transcriber.transcribe_audio(audio_path, language=language)
    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Using mock transcript data to proceed.")
        transcript_data = {
            'transcript': "This is a mock transcript for testing purposes. Hello world.",
            'words': [],
            'language': language,
            'raw_response': "Mock response"
        }


    with open(f"{output_dir}/raw_transcript.json", 'w', encoding='utf-8') as f:
        json.dump({
            'transcript': transcript_data['transcript'],
            'language': transcript_data['language']
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Raw transcript saved: {output_dir}/raw_transcript.json")
    print(f" STranscript length: {len(transcript_data['transcript'])} characters")

    print("\n[2/5] Cleaning transcript...")
    cleaner = TranscriptCleaner()
    clean_text = cleaner.clean_transcript(transcript_data['transcript'])
    segments = cleaner.extract_segments(clean_text, words_per_segment=50)

    with open(f"{output_dir}/clean_transcript.txt", 'w', encoding='utf-8') as f:
        f.write(clean_text)

    with open(f"{output_dir}/segments.json", 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"✓ Clean transcript saved: {output_dir}/clean_transcript.txt")
    print(f"✓ Segments extracted: {len(segments)}")

    print("\n[3/5] Adding noise to audio...")
    noise_injector = NoiseInjector(audio_path)

    noise_configs = [
        ('white_noise_20db', lambda: noise_injector.add_white_noise(snr_db=20)),
        ('white_noise_15db', lambda: noise_injector.add_white_noise(snr_db=15)),
        ('cafe_noise', lambda: noise_injector.add_background_noise('cafe', snr_db=15)),
        ('street_noise', lambda: noise_injector.add_background_noise('street', snr_db=12)),
    ]

    for name, noise_func in noise_configs:
        noisy_audio = noise_func()
        output_path = f"{output_dir}/{name}.wav"
        noise_injector.save_noisy_audio(noisy_audio, output_path)
        print(f"  ✓ Created: {name}.wav")

    print("\n[4/5] Generating training data for reward model...")
    dataset_creator = ASRQualityDataset()
    all_variations = []

    num_segments_to_use = len(segments)
    
    if num_segments_to_use == 0:
        print("Warning: No segments found. Using clean_text to generate data.")
        if clean_text:
            variations = dataset_creator.create_synthetic_pairs(clean_text, num_variations=5)
            all_variations.extend(variations)
        else:
            print("Error: No text available to generate training data.")
            all_variations.extend(dataset_creator.create_synthetic_pairs("Dummy text for training.", num_variations=5))
    else:
        for i, segment in enumerate(segments[:num_segments_to_use]):
            clean_seg_text = cleaner.clean_transcript(segment['text'])
            if not clean_seg_text: continue
            variations = dataset_creator.create_synthetic_pairs(clean_seg_text, num_variations=5)
            all_variations.extend(variations)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{num_segments_to_use} segments...")

    print(f"✓ Generated {len(all_variations)} training examples")

    with open(f"{output_dir}/training_data.json", 'w', encoding='utf-8') as f:
        json.dump(all_variations, f, indent=2, ensure_ascii=False)

    print("\n[5/5] Training reward model...")
    trainer_obj = RewardModelTrainer(model_name="bert-base-multilingual-cased")
    train_loader, val_loader = trainer_obj.prepare_data(all_variations)

    if len(all_variations) == 0:
        print("Error: Cannot train model with no data.")
    else:
        trained_model = trainer_obj.train(
            train_loader,
            val_loader,
            epochs=3,
            output_dir=f"{output_dir}/reward_model"
        )

        print("\n" + "=" * 80)
        print("TESTING REWARD MODEL")
        print("=" * 80)
        
        test_transcripts = []
        if clean_text:
            test_transcripts.append(clean_text[:200])
        if len(all_variations) > 10:
            test_transcripts.append(all_variations[10]['transcript'])
        if len(all_variations) > 40:
            test_transcripts.append(all_variations[40]['transcript'])
        
        if not test_transcripts:
            print("No variations available for testing. Using dummy text.")
            test_transcripts = ["This is perfect.", "this is not so good", "bad text"]

        for i, test_text in enumerate(test_transcripts, 1):
            prediction = trainer_obj.predict_quality(test_text)
            print(f"\nTest {i}:")
            print(f"  Text: {test_text[:100]}...")
            print(f"  Quality: {prediction['quality_label']} (score: {prediction['quality_score']}/5)")
            print(f"  Confidence: {prediction['confidence']:.2%}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  • raw_transcript.json - Original Sarvam AI transcription")
    print("  • clean_transcript.txt - Cleaned transcript")
    print("  • segments.json - Segmented transcript")
    print("  • training_data.json - Training data with quality variations")
    print("  • reward_model/ - Trained reward model")
    print("  • white_noise_*.wav - Noisy audio samples")
    print("  • cafe_noise.wav, street_noise.wav - Background noise samples")
    
    if os.path.exists("dummy_audio.wav"):
        try:
            os.remove("dummy_audio.wav")
            print("\nCleaned up dummy_audio.wav")
        except:
            pass


if __name__ == "__main__":
    load_dotenv()
    
    AUDIO_FILE = "30spod.mp3"
    if not os.path.exists(AUDIO_FILE):
        print(f"Warning: Audio file '{AUDIO_FILE}' not found.")
        print("Creating dummy audio file 'dummy_podcast.wav' for demonstration.")
        sr = 16000
        dummy_audio = np.random.randn(sr * 30)
        sf.write("dummy_podcast.wav", dummy_audio, sr)
        AUDIO_FILE = "dummy_podcast.wav"
        
    
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    
    if not SARVAM_API_KEY:
        print("="*80)
        print("WARNING: 'SARVAM_API_KEY' not found in .env file or environment.")
        print("         The pipeline will run with mock transcription data.")
        print("="*80)
        SARVAM_API_KEY = "sk_mock_key_for_indentation"


    main_pipeline(
        audio_path=AUDIO_FILE,
        sarvam_api_key=SARVAM_API_KEY,
        output_dir="./asr_pipeline_output",
        language="hi-IN"
    )
    
    if AUDIO_FILE == "dummy_podcast.wav" and os.path.exists("dummy_podcast.wav"):
        try:
            os.remove("dummy_podcast.wav")
            print("Cleaned up dummy_podcast.wav")
        except:
            pass
