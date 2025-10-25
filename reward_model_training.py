"""
Reward Model Training Pipeline for ASR Quality Assessment

This script extracts prosody, fluency, and semantic similarity features from audio and transcripts,
implements heuristic rewards like WER, and trains an XGBoost regressor to predict quality scores.
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
from scipy import stats
from scipy.signal import find_peaks
import soundfile as sf

# Text processing and similarity
from sentence_transformers import SentenceTransformer
from jiwer import wer, mer, wil
import re
from collections import Counter

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Progress tracking
from tqdm import tqdm

class AudioFeatureExtractor:
    """Extract prosody and fluency features from audio files"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_prosody_features(self, audio_path: str) -> Dict[str, float]:
        """Extract prosody-related features from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Fundamental frequency (F0) features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            # Remove unvoiced segments
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) == 0:
                return self._get_default_prosody_features()
            
            # F0 statistics
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            f0_range = np.max(f0_voiced) - np.min(f0_voiced)
            f0_median = np.median(f0_voiced)
            
            # F0 contour features
            f0_slope = np.mean(np.diff(f0_voiced))
            f0_curvature = np.mean(np.diff(np.diff(f0_voiced)))
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            energy_range = np.max(rms) - np.min(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Pause detection
            frame_length = 2048
            hop_length = 512
            energy_threshold = 0.01
            silence_frames = rms < energy_threshold
            pause_ratio = np.sum(silence_frames) / len(silence_frames)
            
            # Speaking rate (syllables per second approximation)
            speaking_rate = len(onset_times) / (len(y) / sr)
            
            return {
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'f0_range': f0_range,
                'f0_median': f0_median,
                'f0_slope': f0_slope,
                'f0_curvature': f0_curvature,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'energy_range': energy_range,
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'tempo': tempo,
                'speaking_rate': speaking_rate,
                'pause_ratio': pause_ratio,
                'voiced_ratio': np.sum(voiced_flag) / len(voiced_flag)
            }
            
        except Exception as e:
            print(f"Error extracting prosody features from {audio_path}: {e}")
            return self._get_default_prosody_features()
    
    def extract_fluency_features(self, audio_path: str) -> Dict[str, float]:
        """Extract fluency-related features from audio"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Disfluency detection using energy and spectral features
            rms = librosa.feature.rms(y=y)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Detect potential disfluencies (sudden energy drops/increases)
            energy_diff = np.diff(rms)
            disfluency_candidates = np.abs(energy_diff) > 2 * np.std(energy_diff)
            disfluency_ratio = np.sum(disfluency_candidates) / len(disfluency_candidates)
            
            # Spectral stability
            spectral_stability = 1.0 - np.std(spectral_centroids) / np.mean(spectral_centroids)
            
            # Rhythm regularity
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_intervals = np.diff(onset_frames)
                rhythm_regularity = 1.0 - (np.std(onset_intervals) / np.mean(onset_intervals))
            else:
                rhythm_regularity = 0.0
            
            # Voice quality features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_stability = 1.0 - np.mean(np.std(mfccs, axis=1)) / np.mean(np.abs(mfccs))
            
            return {
                'disfluency_ratio': disfluency_ratio,
                'spectral_stability': spectral_stability,
                'rhythm_regularity': rhythm_regularity,
                'mfcc_stability': mfcc_stability,
                'energy_consistency': 1.0 - (np.std(rms) / np.mean(rms))
            }
            
        except Exception as e:
            print(f"Error extracting fluency features from {audio_path}: {e}")
            return self._get_default_fluency_features()
    
    def _get_default_prosody_features(self) -> Dict[str, float]:
        """Return default prosody features when extraction fails"""
        return {
            'f0_mean': 0.0, 'f0_std': 0.0, 'f0_range': 0.0, 'f0_median': 0.0,
            'f0_slope': 0.0, 'f0_curvature': 0.0, 'energy_mean': 0.0, 'energy_std': 0.0,
            'energy_range': 0.0, 'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'zero_crossing_rate_mean': 0.0, 'tempo': 0.0,
            'speaking_rate': 0.0, 'pause_ratio': 0.0, 'voiced_ratio': 0.0
        }
    
    def _get_default_fluency_features(self) -> Dict[str, float]:
        """Return default fluency features when extraction fails"""
        return {
            'disfluency_ratio': 0.0, 'spectral_stability': 0.0, 'rhythm_regularity': 0.0,
            'mfcc_stability': 0.0, 'energy_consistency': 0.0
        }


class TextFeatureExtractor:
    """Extract semantic similarity and text-based features from transcripts"""
    
    def __init__(self):
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_semantic_features(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Extract semantic similarity features between predicted and reference text"""
        try:
            # Encode texts
            pred_embedding = self.sentence_model.encode([predicted_text])
            ref_embedding = self.sentence_model.encode([reference_text])
            
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
            
            # BLEU-like features (simplified)
            pred_tokens = predicted_text.split()
            ref_tokens = reference_text.split()
            
            # Token overlap
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            token_overlap = len(pred_set.intersection(ref_set)) / max(len(pred_set), len(ref_set), 1)
            
            # Length ratio
            length_ratio = len(pred_tokens) / max(len(ref_tokens), 1)
            
            return {
                'semantic_similarity': semantic_similarity,
                'token_overlap': token_overlap,
                'length_ratio': length_ratio
            }
            
        except Exception as e:
            print(f"Error extracting semantic features: {e}")
            return {'semantic_similarity': 0.0, 'token_overlap': 0.0, 'length_ratio': 1.0}
    
    def extract_text_quality_features(self, text: str) -> Dict[str, float]:
        """Extract text quality features"""
        try:
            # Basic text statistics
            words = text.split()
            chars = len(text)
            word_count = len(words)
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Repetition detection
            word_freq = Counter(words)
            repetition_ratio = sum(1 for count in word_freq.values() if count > 1) / max(word_count, 1)
            
            # Character-level features
            char_diversity = len(set(text)) / max(len(text), 1)
            
            # Punctuation ratio
            punct_chars = sum(1 for c in text if c in '.,!?;:')
            punct_ratio = punct_chars / max(len(text), 1)
            
            # Digit ratio
            digit_chars = sum(1 for c in text if c.isdigit())
            digit_ratio = digit_chars / max(len(text), 1)
            
            return {
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'repetition_ratio': repetition_ratio,
                'char_diversity': char_diversity,
                'punct_ratio': punct_ratio,
                'digit_ratio': digit_ratio
            }
            
        except Exception as e:
            print(f"Error extracting text quality features: {e}")
            return {
                'word_count': 0, 'avg_word_length': 0, 'repetition_ratio': 0,
                'char_diversity': 0, 'punct_ratio': 0, 'digit_ratio': 0
            }


class HeuristicRewardCalculator:
    """Calculate heuristic rewards like WER, MER, WIL"""
    
    def __init__(self):
        pass
    
    def calculate_wer_reward(self, predicted: str, reference: str) -> float:
        """Calculate Word Error Rate (lower is better, so return 1-WER)"""
        try:
            error_rate = wer(reference, predicted)
            return max(0.0, 1.0 - error_rate)
        except:
            return 0.0
    
    def calculate_mer_reward(self, predicted: str, reference: str) -> float:
        """Calculate Match Error Rate (lower is better)"""
        try:
            error_rate = mer(reference, predicted)
            return max(0.0, 1.0 - error_rate)
        except:
            return 0.0
    
    def calculate_wil_reward(self, predicted: str, reference: str) -> float:
        """Calculate Word Information Lost (lower is better)"""
        try:
            error_rate = wil(reference, predicted)
            return max(0.0, 1.0 - error_rate)
        except:
            return 0.0
    
    def calculate_length_penalty(self, predicted: str, reference: str) -> float:
        """Calculate length penalty (penalize if too different)"""
        try:
            pred_len = len(predicted.split())
            ref_len = len(reference.split())
            if ref_len == 0:
                return 0.0
            ratio = pred_len / ref_len
            # Penalty if ratio is too far from 1.0
            penalty = 1.0 - abs(ratio - 1.0)
            return max(0.0, penalty)
        except:
            return 0.0
    
    def calculate_quality_score(self, predicted: str, reference: str) -> float:
        """Calculate overall quality score combining multiple metrics"""
        wer_score = self.calculate_wer_reward(predicted, reference)
        mer_score = self.calculate_mer_reward(predicted, reference)
        wil_score = self.calculate_wil_reward(predicted, reference)
        length_penalty = self.calculate_length_penalty(predicted, reference)
        
        # Weighted combination
        quality_score = (0.4 * wer_score + 0.3 * mer_score + 0.2 * wil_score + 0.1 * length_penalty)
        return quality_score


class RewardModelTrainer:
    """Train XGBoost regressor for reward prediction"""
    
    def __init__(self):
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.reward_calculator = HeuristicRewardCalculator()
        self.scaler = StandardScaler()
        self.model = None
        
    def extract_all_features(self, audio_path: str, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Extract all features for a single sample"""
        features = {}
        
        # Audio features
        prosody_features = self.audio_extractor.extract_prosody_features(audio_path)
        fluency_features = self.audio_extractor.extract_fluency_features(audio_path)
        
        # Text features
        semantic_features = self.text_extractor.extract_semantic_features(predicted_text, reference_text)
        text_quality_features = self.text_extractor.extract_text_quality_features(predicted_text)
        
        # Heuristic rewards
        wer_reward = self.reward_calculator.calculate_wer_reward(predicted_text, reference_text)
        mer_reward = self.reward_calculator.calculate_mer_reward(predicted_text, reference_text)
        wil_reward = self.reward_calculator.calculate_wil_reward(predicted_text, reference_text)
        length_penalty = self.reward_calculator.calculate_length_penalty(predicted_text, reference_text)
        quality_score = self.reward_calculator.calculate_quality_score(predicted_text, reference_text)
        
        # Combine all features
        features.update(prosody_features)
        features.update(fluency_features)
        features.update(semantic_features)
        features.update(text_quality_features)
        features.update({
            'wer_reward': wer_reward,
            'mer_reward': mer_reward,
            'wil_reward': wil_reward,
            'length_penalty': length_penalty,
            'quality_score': quality_score
        })
        
        return features
    
    def prepare_training_data(self, training_data_path: str, audio_dir: str = ".") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from JSON file"""
        print("Loading training data...")
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        features_list = []
        targets = []
        
        print(f"Processing {len(training_data)} samples...")
        
        for i, sample in enumerate(tqdm(training_data, desc="Extracting features")):
            try:
                # Get transcript and quality score
                transcript = sample['transcript']
                quality_score = sample['quality_score']
                
                # For this example, we'll use the transcript as both predicted and reference
                # In a real scenario, you'd have separate predicted and reference texts
                predicted_text = transcript
                reference_text = transcript  # Using same as reference for now
                
                # Find corresponding audio file (assuming naming convention)
                # You may need to adjust this based on your file structure
                audio_file = None
                for ext in ['.mp3', '.wav', '.m4a']:
                    potential_file = os.path.join(audio_dir, f"sample_{i}{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                
                if audio_file is None:
                    print(f"Warning: No audio file found for sample {i}")
                    # Use default features
                    features = self._get_default_features()
                else:
                    features = self.extract_all_features(audio_file, predicted_text, reference_text)
                
                features_list.append(features)
                targets.append(quality_score)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use default features
                features = self._get_default_features()
                features_list.append(features)
                targets.append(1.0)  # Default target
        
        # Convert to arrays
        feature_names = list(features_list[0].keys())
        X = np.array([[features[name] for name in feature_names] for features in features_list])
        y = np.array(targets)
        
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X, y, feature_names
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when extraction fails"""
        default_features = {}
        
        # Prosody features
        prosody_defaults = self.audio_extractor._get_default_prosody_features()
        default_features.update(prosody_defaults)
        
        # Fluency features
        fluency_defaults = self.audio_extractor._get_default_fluency_features()
        default_features.update(fluency_defaults)
        
        # Semantic features
        default_features.update({'semantic_similarity': 0.0, 'token_overlap': 0.0, 'length_ratio': 1.0})
        
        # Text quality features
        default_features.update({
            'word_count': 0, 'avg_word_length': 0, 'repetition_ratio': 0,
            'char_diversity': 0, 'punct_ratio': 0, 'digit_ratio': 0
        })
        
        # Heuristic rewards
        default_features.update({
            'wer_reward': 0.0, 'mer_reward': 0.0, 'wil_reward': 0.0,
            'length_penalty': 0.0, 'quality_score': 0.0
        })
        
        return default_features
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """Train XGBoost regressor"""
        print("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"\nCross-validation RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std() * 2):.4f})")
    
    def save_model(self, model_path: str = "reward_model.pkl") -> None:
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.audio_extractor.__class__.__name__ + "_features"
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "reward_model.pkl") -> None:
        """Load trained model and scaler"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"Model loaded from {model_path}")
    
    def predict_reward(self, audio_path: str, predicted_text: str, reference_text: str) -> float:
        """Predict reward score for new sample"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")
        
        features = self.extract_all_features(audio_path, predicted_text, reference_text)
        feature_names = list(features.keys())
        X = np.array([[features[name] for name in feature_names]]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        reward_score = self.model.predict(X_scaled)[0]
        return reward_score


def main():
    """Main execution function"""
    print("Starting Reward Model Training Pipeline...")
    
    # Initialize trainer
    trainer = RewardModelTrainer()
    
    # Prepare training data
    training_data_path = "asr_pipeline_output/training_data.json"
    audio_dir = "."  # Adjust based on your audio file location
    
    if not os.path.exists(training_data_path):
        print(f"Training data file not found: {training_data_path}")
        return
    
    try:
        # Extract features and prepare data
        X, y, feature_names = trainer.prepare_training_data(training_data_path, audio_dir)
        
        # Train model
        trainer.train_model(X, y, feature_names)
        
        # Save model
        trainer.save_model("reward_model.pkl")
        
        print("\nTraining completed successfully!")
        print("Model saved as 'reward_model.pkl'")
        
        # Example prediction (if you have test audio)
        # reward_score = trainer.predict_reward("path/to/audio.wav", "predicted text", "reference text")
        # print(f"Predicted reward score: {reward_score}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
