"""
Example script demonstrating ASR reranking using the lightweight reward model

This script shows how to use the lightweight reward model to rerank multiple
ASR outputs from Sarvam STT for better quality.
"""

import os
import json
import numpy as np
from main2 import LightweightRewardModel, create_training_data_from_sarvam_output


def create_sample_transcripts():
    """Create sample transcripts for demonstration"""
    return [
        {
            'transcript': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।',
            'transcript_data': {
                'request_id': 'sample_1',
                'language_code': 'hi-IN'
            }
        },
        {
            'transcript': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी। शायद सेंसेक्स की एक्सपायरी थी।',
            'transcript_data': {
                'request_id': 'sample_2',
                'language_code': 'hi-IN'
            }
        },
        {
            'transcript': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी। शायद सेंसेक्स की एक्सपायरी थी। तो उन्होंने स्क्रीन पर देखा कि उनकी जो 10 12 15 या 20 या 50 जो भी उन्होंने बेचा था वो चार पांच रुपए की रह गई।',
            'transcript_data': {
                'request_id': 'sample_3',
                'language_code': 'hi-IN'
            }
        }
    ]


def demonstrate_feature_extraction():
    """Demonstrate feature extraction"""
    print("=== Feature Extraction Demo ===")
    
    # Initialize feature extractor
    from main2 import LightweightFeatureExtractor
    extractor = LightweightFeatureExtractor()
    
    # Sample data
    predicted_text = "मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी।"
    reference_text = "मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।"
    transcript_data = {'request_id': 'demo', 'language_code': 'hi-IN'}
    
    # Extract features
    features = extractor.extract_all_features(predicted_text, reference_text, transcript_data)
    
    print(f"Extracted {len(features)} features:")
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.4f}")
    
    return features


def demonstrate_training():
    """Demonstrate model training"""
    print("\n=== Model Training Demo ===")
    
    # Initialize model
    reward_model = LightweightRewardModel()
    
    # Create sample training data
    training_data = [
        {
            'transcript': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी।',
            'reference': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।',
            'transcript_data': {'request_id': 'train_1'},
            'quality_score': 4.0
        },
        {
            'transcript': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।',
            'reference': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।',
            'transcript_data': {'request_id': 'train_2'},
            'quality_score': 5.0
        },
        {
            'transcript': 'मतलब एक भैया थे।',
            'reference': 'मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।',
            'transcript_data': {'request_id': 'train_3'},
            'quality_score': 2.0
        }
    ]
    
    # Prepare training data
    X, y = reward_model.prepare_training_data(training_data)
    
    # Train model
    reward_model.train_model(X, y)
    
    # Save model
    reward_model.save_model("demo_lightweight_model.pkl")
    
    return reward_model


def demonstrate_reranking():
    """Demonstrate transcript reranking"""
    print("\n=== Reranking Demo ===")
    
    # Load trained model
    reward_model = LightweightRewardModel()
    try:
        reward_model.load_model("demo_lightweight_model.pkl")
    except:
        print("Model not found, training first...")
        reward_model = demonstrate_training()
    
    # Create sample transcripts to rerank
    transcripts = create_sample_transcripts()
    reference_text = "मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी। शायद सेंसेक्स की एक्सपायरी थी। तो उन्होंने स्क्रीन पर देखा कि उनकी जो 10 12 15 या 20 या 50 जो भी उन्होंने बेचा था वो चार पांच रुपए की रह गई।"
    
    print("Original transcripts:")
    for i, transcript in enumerate(transcripts):
        print(f"{i+1}. {transcript['transcript'][:50]}...")
    
    # Rerank transcripts
    reranked_transcripts = reward_model.rerank_transcripts(transcripts, reference_text)
    
    print("\nReranked transcripts (by predicted quality score):")
    for i, transcript in enumerate(reranked_transcripts):
        score = transcript.get('predicted_score', 0.0)
        text = transcript['transcript'][:50]
        print(f"{i+1}. Score: {score:.4f} - {text}...")
    
    return reranked_transcripts


def demonstrate_single_prediction():
    """Demonstrate single transcript scoring"""
    print("\n=== Single Prediction Demo ===")
    
    # Load trained model
    reward_model = LightweightRewardModel()
    try:
        reward_model.load_model("demo_lightweight_model.pkl")
    except:
        print("Model not found, training first...")
        reward_model = demonstrate_training()
    
    # Sample data
    predicted_text = "मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी।"
    reference_text = "मतलब एक भैया थे। वो उनको उन्होंने देखा कुछ एक्सपायरी थी मुझे नहीं पता वो किस दिन की एक्सपायरी थी।"
    transcript_data = {'request_id': 'single_pred', 'language_code': 'hi-IN'}
    
    # Predict score
    score = reward_model.predict_score(predicted_text, reference_text, transcript_data)
    
    print(f"Predicted quality score: {score:.4f}")
    print(f"Predicted text: {predicted_text}")
    print(f"Reference text: {reference_text}")
    
    return score


def demonstrate_with_real_data():
    """Demonstrate with real Sarvam data if available"""
    print("\n=== Real Data Demo ===")
    
    sarvam_output_path = "batch_output/10minpod.mp3.json"
    reference_path = "hindi_corrected_transcript.txt"
    
    if os.path.exists(sarvam_output_path) and os.path.exists(reference_path):
        print("Using real Sarvam data...")
        
        # Create training data from real files
        training_data = create_training_data_from_sarvam_output(sarvam_output_path, reference_path)
        
        # Initialize and train model
        reward_model = LightweightRewardModel()
        X, y = reward_model.prepare_training_data(training_data)
        reward_model.train_model(X, y)
        
        # Create multiple versions of the transcript for reranking
        with open(sarvam_output_path, 'r', encoding='utf-8') as f:
            sarvam_data = json.load(f)
        
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
        
        # Create different versions of the transcript
        original_transcript = sarvam_data.get('transcript', '')
        
        # Create variations (simulating different ASR outputs)
        variations = [
            {
                'transcript': original_transcript,
                'transcript_data': sarvam_data,
                'version': 'original'
            },
            {
                'transcript': original_transcript[:len(original_transcript)//2],  # Half length
                'transcript_data': sarvam_data,
                'version': 'truncated'
            },
            {
                'transcript': original_transcript + " अतिरिक्त पाठ।",  # With extra text
                'transcript_data': sarvam_data,
                'version': 'extended'
            }
        ]
        
        # Rerank variations
        reranked = reward_model.rerank_transcripts(variations, reference_text)
        
        print("Reranked real data variations:")
        for i, transcript in enumerate(reranked):
            score = transcript.get('predicted_score', 0.0)
            version = transcript.get('version', 'unknown')
            text_preview = transcript['transcript'][:100]
            print(f"{i+1}. Score: {score:.4f} ({version}) - {text_preview}...")
        
        return reranked
    else:
        print("Real data files not found, skipping real data demo")
        return None


def main():
    """Main demonstration function"""
    print("ASR Reranking with Lightweight Reward Model")
    print("=" * 50)
    
    try:
        # 1. Feature extraction demo
        features = demonstrate_feature_extraction()
        
        # 2. Training demo
        model = demonstrate_training()
        
        # 3. Single prediction demo
        score = demonstrate_single_prediction()
        
        # 4. Reranking demo
        reranked = demonstrate_reranking()
        
        # 5. Real data demo (if available)
        real_reranked = demonstrate_with_real_data()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("The lightweight reward model is ready for ASR reranking.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
