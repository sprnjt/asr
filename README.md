# Gemini-Powered ASR Reward Model

This project implements a sophisticated, intelligent reward model for evaluating Automatic Speech Recognition (ASR) transcriptions. It moves beyond traditional Word Error Rate (WER) by using a Large Language Model (Gemini 2.0 Flash) to perform a human-like, context-aware analysis.

## The Problem with Standard WER

The primary problem with standard WER is its inability to distinguish between trivial and critical errors:

This model addresses these shortcomings by calculating a weighted "reward score" (0.0 to 1.0) based on a deep understanding of the text, with a special focus on high-stakes domains like finance and stock trading.

## Intelligent ASR Evaluation: Six Key Components

This model computes a final reward score by analyzing six key components of transcription quality:

### 1. Number Normalization (LLM-Powered)

**Problem:** ASR models often transcribe numbers in various formats (e.g., "10", "ten", "das", "दस"). Traditional rule-based, dictionary-lookup normalization is brittle and fails to scale.

**Our Solution:** We use Gemini to perform intelligent, context-aware number normalization before calculating WER. It understands Hindi, English, and Hinglish transliterations, ensuring that "das" and "10" are treated as identical.

### 2. Proper Noun Accuracy (LLM-Powered)

**Problem:** In finance, proper nouns (company names, stock tickers) are critical. A rule-based system (dictionaries, regex) cannot differentiate a major error (HDFC vs. HSBC) from a minor variation (BSE vs. Bombay Stock Exchange).

**Our Solution:** The model uses Gemini to extract all proper nouns (Organizations, Tickers, Products) and then semantically matches them. It correctly identifies HDFC vs. HSBC as a mismatch while understanding that BSE and Bombay Stock Exchange refer to the same entity. This component is heavily weighted (40%) in the final score.

### 3. Semantic Similarity (LLM-Powered)

**Problem:** Two sentences can have identical meanings but different words (e.g., "turn off the lamp" vs. "switch off the light"). Simple word-overlap metrics (like Jaccard) or even standard sentence-transformer models can fail to capture the full semantic intent.

**Our Solution:** We leverage Gemini to provide a 0.0-1.0 semantic similarity score that assesses if the core intent of the reference and hypothesis are the same.

### 4. Repetitive Word Penalty

**Problem:** ASR models sometimes get "stuck" in a repetitive loop (e.g., "hello hello hello hello").

**Our Solution:** We implement a MalformedTextDetector that penalizes this "stuttering." The penalty is higher if a larger fraction of the text consists of these repetitions, cleaning the output for downstream tasks.

### 5. Speaker Diarization Accuracy

**Problem:** In a conversation, knowing who spoke is as important as what was said.

**Our Solution:** The model parses speaker labels (e.g., ss:, ps:) from both the reference and hypothesis. It then compares the segments to calculate speaker accuracy, ensuring the correct speaker is attributed to the correct text.

### 6. Smarter Word Error Rate (WER)

**Problem:** Standard WER is "dumb" and penalizes trivial errors heavily.

**Our Solution:** Our WER calculation is "smarter" because it runs after the Gemini-powered number normalization. This means the WER metric focuses on actual word errors, not simple formatting differences.

## Architecture

The evaluation pipeline is orchestrated by the `RewardModelTrainer` class:

1. **Input:** The system takes a `ground_truth.json` file (with audio paths and reference text) and API keys.

2. **ASR Transcription:** (In evaluate mode) The `SarvamAITranscriber` transcribes the audio file to get the hypothesis text.

3. **Feature Extraction:** The `extract_features` method runs all scorers:
   - `GeminiNumberNormalizer`: Extracts and normalizes all numbers.
   - `GeminiProperNounExtractor`: Extracts, categorizes, and matches all proper nouns.
   - `WERCalculator`: Calculates WER on the normalized text.
   - `SemanticSimilarityScorer`: Gets a 0.0-1.0 semantic score from Gemini.
   - `SpeakerDiarizationScorer`: Calculates speaker label accuracy.
   - `MalformedTextDetector`: Calculates a repetition penalty.

4. **Reward Score Calculation:** The `calculate_reward_score` method combines all these features using a weighted average to produce the final score from 0.0 to 1.0.

5. **Output:** The script saves a detailed `evaluation_results.json` and a `reward_model_gemini.pkl` file with the model's weights.


## Usage

The script is run from the command line and has two modes: `test` and `evaluate`.

### Test Mode

This mode runs a hard-coded example (HDFC vs. HSBC) to quickly verify that the Gemini API is working and the scoring logic is correct.

```bash
python finalfinal.py \
    --mode test \
    --gemini_api_key "YOUR_GEMINI_API_KEY"
```

### Evaluate Mode

This mode runs the full pipeline. It reads your `ground_truth.json`, transcribes each audio file using the Sarvam AI API, and then scores the transcription against the reference text using the Gemini-powered reward model.

```bash
python finalfinal.py \
    --mode evaluate \
    --gemini_api_key "YOUR_GEMINI_API_KEY" \
    --sarvam_api_key "YOUR_SARVAM_AI_API_KEY" \
    --ground_truth "path/to/ground_truth.json" \
    --output "results/evaluation_results.json"
```

## Command-Line Arguments

- `--mode`: `test` or `evaluate` (default: `test`)
- `--gemini_api_key`: (Required) Your Google AI Studio API key for Gemini
- `--sarvam_api_key`: (Required for evaluate mode) Your API key for the Sarvam AI ASR service
- `--ground_truth`: Path to the ground truth JSON file (default: `ground_truth.json`)
- `--model`: Path to save the output model pickle file (default: `reward_model_gemini.pkl`)
- `--output`: Path to save the JSON results file (default: `evaluation_results.json`)

## Configuration

### Ground Truth JSON Format

The `--ground_truth` file must be a JSON list, where each object contains a path to an audio file and the corresponding reference transcription.

Example `ground_truth.json`:

```json
[
  {
    "audio_path": "data/audio/sample_001.wav",
    "reference_text": "ss: Hi Hello\nps: How are You\nss: Mereko 10 share HDFC ke chahiye at 100.34\nps: Hojayega logging into system"
  },
  {
    "audio_path": "data/audio/sample_002.wav",
    "reference_text": "ps: Yes sir, I am booking the trade for 50 shares of Reliance at market price."
  }
]
```

## Outputs

The evaluate mode generates two main files:

1. **`evaluation_results.json`**: A detailed JSON report for every file processed, including the reference, hypothesis, all calculated features, and the final reward score.

2. **`reward_model_gemini.pkl`**: A pickle file containing the model's configuration and scoring weights.

## Scoring Weights

The final reward score is calculated using the following weights:

- **Proper Noun Score**: 40% (highest weight due to critical importance in finance)
- **WER Component**: 15%
- **Semantic Similarity**: 15%
- **Repetition Penalty**: 10%
- **Speaker Accuracy**: 10%
- **Number Accuracy**: 10%

## Example Output

```
FINAL REWARD SCORE: 0.7234

Score Breakdown:
  WER Component: 0.1275 (15%)
  Proper Noun Component: 0.2894 (40%, Gemini-matched)
  Repetition Component: 0.0950 (10%)
  Semantic Component: 0.1085 (15%, Gemini)
  Speaker Component: 0.0950 (10%)
  Number Component: 0.0950 (10%, Gemini)
```

## API Requirements

- **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/)
- **Sarvam AI API Key**: Get from [Sarvam AI](https://sarvam.ai/) (for evaluate mode)
