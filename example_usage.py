"""
Example usage of the ASR Reward Model Training Pipeline

This script demonstrates how to use the reward model training and prediction system.
"""

import os
import sys
from main import train_reward_model, predict_quality

def example_training():
    """Example of training the reward model"""
    print("=== Training Example ===")
    
    # Check if training data exists
    training_data_path = "asr_pipeline_output/training_data.json"
    if not os.path.exists(training_data_path):
        print(f"Training data not found at {training_data_path}")
        print("Please ensure you have the training data file.")
        return False
    
    # Train the model
    success = train_reward_model(
        training_data_path=training_data_path,
        audio_dir=".",  # Directory containing audio files
        model_output_path="reward_model.pkl"
    )
    
    if success:
        print("Training completed successfully!")
        return True
    else:
        print("Training failed!")
        return False

def example_prediction():
    """Example of using the trained model for prediction"""
    print("=== Prediction Example ===")
    
    # Check if model exists
    model_path = "reward_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using example_training()")
        return False
    
    # Example audio and text (you would replace these with actual files)
    audio_path = "10minpod.mp3"  # Replace with your audio file
    predicted_text = "This is a sample predicted transcript"
    reference_text = "This is a sample reference transcript"
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        print("Please provide a valid audio file path")
        return False
    
    # Predict quality score
    score = predict_quality(
        audio_path=audio_path,
        predicted_text=predicted_text,
        reference_text=reference_text,
        model_path=model_path
    )
    
    if score is not None:
        print(f"Predicted quality score: {score:.4f}")
        return True
    else:
        print("Prediction failed!")
        return False

def main():
    """Main example function"""
    print("ASR Reward Model Training - Example Usage")
    print("=" * 50)
    
    # Example 1: Training
    print("\n1. Training the reward model...")
    training_success = example_training()
    
    if training_success:
        print("\n2. Using the trained model for prediction...")
        prediction_success = example_prediction()
        
        if prediction_success:
            print("\nExample completed successfully!")
        else:
            print("\nPrediction example failed!")
    else:
        print("\nTraining example failed!")
        print("Please check your training data and try again.")

if __name__ == "__main__":
    main()
