"""
Lightweight Reward Model for ASR Reranking

This script implements a lightweight reward model that scores ASR outputs using
WER, confidence, and semantic similarity features to rerank Sarvam STT transcripts
for better ASR quality.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Text processing and similarity
from sentence_transformers import SentenceTransformer
from jiwer import wer, mer, wil
import re
from collections import Counter

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Progress tracking
from tqdm import tqdm


class LightweightFeatureExtractor:
    """Extract lightweight features for ASR reranking"""
    
    def __init__(self):
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_wer_features(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Extract WER-based features"""
        try:
            # Calculate WER metrics
            wer_score = wer(reference_text, predicted_text)
            mer_score = mer(reference_text, predicted_text)
            wil_score = wil(reference_text, predicted_text)
            
            # Convert to rewards (lower is better, so 1 - error_rate)
            wer_reward = max(0.0, 1.0 - wer_score)
            mer_reward = max(0.0, 1.0 - mer_score)
            wil_reward = max(0.0, 1.0 - wil_score)
            
            # Length penalty
            pred_len = len(predicted_text.split())
            ref_len = len(reference_text.split())
            if ref_len == 0:
                length_penalty = 0.0
            else:
                length_ratio = pred_len / ref_len
                # Penalty if ratio is too far from 1.0
                length_penalty = 1.0 - abs(length_ratio - 1.0)
                length_penalty = max(0.0, length_penalty)
            
            return {
                'wer_score': wer_score,
                'wer_reward': wer_reward,
                'mer_score': mer_score,
                'mer_reward': mer_reward,
                'wil_score': wil_score,
                'wil_reward': wil_reward,
                'length_penalty': length_penalty,
                'length_ratio': pred_len / max(ref_len, 1)
            }
            
        except Exception as e:
            print(f"Error calculating WER features: {e}")
            return {
                'wer_score': 1.0, 'wer_reward': 0.0, 'mer_score': 1.0, 'mer_reward': 0.0,
                'wil_score': 1.0, 'wil_reward': 0.0, 'length_penalty': 0.0, 'length_ratio': 1.0
            }
    
    def extract_confidence_features(self, transcript_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence-related features from Sarvam STT output"""
        try:
            # Extract confidence scores if available
            confidence_scores = []
            
            # Check for word-level confidence scores
            if 'timestamps' in transcript_data and 'words' in transcript_data['timestamps']:
                words = transcript_data['timestamps']['words']
                # If confidence scores are available in the data structure
                if isinstance(words, list) and len(words) > 0:
                    # Try to extract confidence from word-level data
                    for word_data in words:
                        if isinstance(word_data, dict) and 'confidence' in word_data:
                            confidence_scores.append(word_data['confidence'])
                        elif isinstance(word_data, str):
                            # If no confidence data, use default
                            confidence_scores.append(0.8)  # Default confidence
            
            # If no confidence scores found, use default
            if not confidence_scores:
                confidence_scores = [0.8]  # Default confidence
            
            # Calculate confidence statistics
            mean_confidence = np.mean(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)
            std_confidence = np.std(confidence_scores)
            
            # Low confidence ratio (scores below threshold)
            low_confidence_ratio = sum(1 for c in confidence_scores if c < 0.5) / len(confidence_scores)
            
            # High confidence ratio (scores above threshold)
            high_confidence_ratio = sum(1 for c in confidence_scores if c > 0.8) / len(confidence_scores)
            
            return {
                'mean_confidence': mean_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'std_confidence': std_confidence,
                'low_confidence_ratio': low_confidence_ratio,
                'high_confidence_ratio': high_confidence_ratio
            }
            
        except Exception as e:
            print(f"Error extracting confidence features: {e}")
            return {
                'mean_confidence': 0.8, 'min_confidence': 0.8, 'max_confidence': 0.8,
                'std_confidence': 0.0, 'low_confidence_ratio': 0.0, 'high_confidence_ratio': 1.0
            }
    
    def extract_semantic_features(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Extract semantic similarity features"""
        try:
            # Encode texts
            pred_embedding = self.sentence_model.encode([predicted_text])
            ref_embedding = self.sentence_model.encode([reference_text])
            
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
            
            # Token-level features
            pred_tokens = predicted_text.split()
            ref_tokens = reference_text.split()
            
            # Token overlap
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            token_overlap = len(pred_set.intersection(ref_set)) / max(len(pred_set), len(ref_set), 1)
            
            # Jaccard similarity
            intersection = len(pred_set.intersection(ref_set))
            union = len(pred_set.union(ref_set))
            jaccard_similarity = intersection / max(union, 1)
            
            return {
                'semantic_similarity': semantic_similarity,
                'token_overlap': token_overlap,
                'jaccard_similarity': jaccard_similarity
            }
            
        except Exception as e:
            print(f"Error extracting semantic features: {e}")
            return {
                'semantic_similarity': 0.0, 'token_overlap': 0.0, 'jaccard_similarity': 0.0
            }
    
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
            
            # Special character ratio (non-alphanumeric)
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / max(len(text), 1)
            
            return {
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'repetition_ratio': repetition_ratio,
                'char_diversity': char_diversity,
                'punct_ratio': punct_ratio,
                'digit_ratio': digit_ratio,
                'special_ratio': special_ratio
            }
            
        except Exception as e:
            print(f"Error extracting text quality features: {e}")
            return {
                'word_count': 0, 'avg_word_length': 0, 'repetition_ratio': 0,
                'char_diversity': 0, 'punct_ratio': 0, 'digit_ratio': 0, 'special_ratio': 0
            }
    
    def extract_all_features(self, predicted_text: str, reference_text: str, 
                           transcript_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract all lightweight features"""
        features = {}
        
        # WER features
        wer_features = self.extract_wer_features(predicted_text, reference_text)
        features.update(wer_features)
        
        # Confidence features
        if transcript_data is None:
            transcript_data = {}
        confidence_features = self.extract_confidence_features(transcript_data)
        features.update(confidence_features)
        
        # Semantic features
        semantic_features = self.extract_semantic_features(predicted_text, reference_text)
        features.update(semantic_features)
        
        # Text quality features
        text_quality_features = self.extract_text_quality_features(predicted_text)
        features.update(text_quality_features)
        
        return features


class LightweightRewardModel:
    """Lightweight reward model for ASR reranking"""
    
    def __init__(self):
        self.feature_extractor = LightweightFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from list of samples"""
        print(f"Processing {len(training_data)} training samples...")
        
        features_list = []
        targets = []
        
        for i, sample in enumerate(tqdm(training_data, desc="Extracting features")):
            try:
                # Extract features
                predicted_text = sample.get('transcript', '')
                reference_text = sample.get('reference', sample.get('transcript', ''))
                transcript_data = sample.get('transcript_data', {})
                quality_score = sample.get('quality_score', 1.0)
                
                features = self.feature_extractor.extract_all_features(
                    predicted_text, reference_text, transcript_data
                )
                
                features_list.append(features)
                targets.append(quality_score)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use default features
                features = self._get_default_features()
                features_list.append(features)
                targets.append(1.0)
        
        # Convert to arrays
        if not features_list:
            raise ValueError("No features extracted!")
        
        self.feature_names = list(features_list[0].keys())
        X = np.array([[features[name] for name in self.feature_names] for features in features_list])
        y = np.array(targets)
        
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X, y
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when extraction fails"""
        return {
            # WER features
            'wer_score': 1.0, 'wer_reward': 0.0, 'mer_score': 1.0, 'mer_reward': 0.0,
            'wil_score': 1.0, 'wil_reward': 0.0, 'length_penalty': 0.0, 'length_ratio': 1.0,
            # Confidence features
            'mean_confidence': 0.8, 'min_confidence': 0.8, 'max_confidence': 0.8,
            'std_confidence': 0.0, 'low_confidence_ratio': 0.0, 'high_confidence_ratio': 1.0,
            # Semantic features
            'semantic_similarity': 0.0, 'token_overlap': 0.0, 'jaccard_similarity': 0.0,
            # Text quality features
            'word_count': 0, 'avg_word_length': 0, 'repetition_ratio': 0,
            'char_diversity': 0, 'punct_ratio': 0, 'digit_ratio': 0, 'special_ratio': 0
        }
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the lightweight reward model"""
        print("Training lightweight reward model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model (lightweight and effective)
        self.model = RandomForestRegressor(
            n_estimators=50,  # Reduced for lightweight model
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
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
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"\nCross-validation RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(cv_scores.std() * 2):.4f})")
    
    def predict_score(self, predicted_text: str, reference_text: str, 
                     transcript_data: Optional[Dict[str, Any]] = None) -> float:
        """Predict quality score for a single transcript"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        features = self.feature_extractor.extract_all_features(
            predicted_text, reference_text, transcript_data
        )
        
        X = np.array([[features[name] for name in self.feature_names]]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        score = self.model.predict(X_scaled)[0]
        return score
    
    def rerank_transcripts(self, transcripts: List[Dict[str, Any]], 
                          reference_text: str) -> List[Dict[str, Any]]:
        """Rerank multiple transcripts based on predicted quality scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        scored_transcripts = []
        
        for transcript in transcripts:
            try:
                predicted_text = transcript.get('transcript', '')
                transcript_data = transcript.get('transcript_data', {})
                
                score = self.predict_score(predicted_text, reference_text, transcript_data)
                
                # Add score to transcript data
                transcript_with_score = transcript.copy()
                transcript_with_score['predicted_score'] = score
                scored_transcripts.append(transcript_with_score)
                
            except Exception as e:
                print(f"Error scoring transcript: {e}")
                # Add default score
                transcript_with_score = transcript.copy()
                transcript_with_score['predicted_score'] = 0.0
                scored_transcripts.append(transcript_with_score)
        
        # Sort by predicted score (descending)
        reranked_transcripts = sorted(scored_transcripts, key=lambda x: x['predicted_score'], reverse=True)
        
        return reranked_transcripts
    
    def save_model(self, model_path: str = "lightweight_reward_model.pkl") -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "lightweight_reward_model.pkl") -> None:
        """Load trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")


def create_training_data_from_sarvam_output(sarvam_output_path: str, 
                                           reference_path: str) -> List[Dict[str, Any]]:
    """Create training data from Sarvam STT output and reference text"""
    print("Creating training data from Sarvam output...")
    
    # Load Sarvam output
    with open(sarvam_output_path, 'r', encoding='utf-8') as f:
        sarvam_data = json.load(f)
    
    # Load reference text
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_text = f.read().strip()
    
    # Create training samples
    training_data = []
    
    # Use the main transcript as predicted text
    predicted_text = sarvam_data.get('transcript', '')
    
    # Create a sample with quality score based on WER
    try:
        from jiwer import wer
        wer_score = wer(reference_text, predicted_text)
        # Convert WER to quality score (1-5 scale)
        if wer_score < 0.1:
            quality_score = 5  # Excellent
        elif wer_score < 0.2:
            quality_score = 4  # Good
        elif wer_score < 0.3:
            quality_score = 3  # Fair
        elif wer_score < 0.5:
            quality_score = 2  # Poor
        else:
            quality_score = 1  # Very Poor
    except:
        quality_score = 3  # Default
    
    sample = {
        'transcript': predicted_text,
        'reference': reference_text,
        'transcript_data': sarvam_data,
        'quality_score': quality_score
    }
    
    training_data.append(sample)
    
    print(f"Created {len(training_data)} training samples")
    return training_data


def main():
    """Main execution function"""
    print("Lightweight Reward Model for ASR Reranking")
    print("=" * 50)
    
    # Initialize model
    reward_model = LightweightRewardModel()
    
    # Create training data from existing Sarvam output
    sarvam_output_path = "batch_output/10minpod.mp3.json"
    reference_path = "hindi_corrected_transcript.txt"
    
    if os.path.exists(sarvam_output_path) and os.path.exists(reference_path):
        print("Creating training data from Sarvam output...")
        training_data = create_training_data_from_sarvam_output(sarvam_output_path, reference_path)
        
        # Prepare training data
        X, y = reward_model.prepare_training_data(training_data)
        
        # Train model
        reward_model.train_model(X, y)
        
        # Save model
        reward_model.save_model("lightweight_reward_model.pkl")
        
        print("\nTraining completed successfully!")
        
        # Example reranking
        print("\nExample: Reranking transcripts...")
        transcripts = [
            {'transcript': 'Sample transcript 1', 'transcript_data': {}},
            {'transcript': 'Sample transcript 2', 'transcript_data': {}},
            {'transcript': 'Sample transcript 3', 'transcript_data': {}}
        ]
        
        reference = "Reference text for comparison"
        reranked = reward_model.rerank_transcripts(transcripts, reference)
        
        print("Reranked transcripts:")
        for i, transcript in enumerate(reranked):
            print(f"{i+1}. Score: {transcript['predicted_score']:.4f} - {transcript['transcript'][:50]}...")
        
    else:
        print("Required files not found:")
        print(f"  - Sarvam output: {sarvam_output_path}")
        print(f"  - Reference text: {reference_path}")
        print("Please ensure these files exist for training.")


if __name__ == "__main__":
    main()
