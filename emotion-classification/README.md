# Emotion Classification using MiniLM (Hugging Face Transformers)

This project fine-tunes the MiniLM transformer model on the Hugging Face Emotion dataset to classify text into 6 emotion categories.

---

## Model

- Base model: microsoft/MiniLM-L12-H384-uncased
- Task: Text Classification
- Architecture: Transformer Encoder (MiniLM)
- Labels:
  - sadness
  - joy
  - love
  - anger
  - fear
  - surprise

---

## Dataset

Hugging Face Emotion Dataset  
https://huggingface.co/datasets/emotion

Dataset contains:

- Train samples: 16,000
- Validation samples: 2,000
- Test samples: 2,000

Example:

```
Text: "I am feeling very happy today"
Label: joy
```

---

## Features

- Fine-tuning MiniLM transformer
- Custom Trainer with weighted CrossEntropyLoss
- Handles class imbalance
- Uses Hugging Face Trainer API
- Push trained model to Hugging Face Hub
- Inference using pipeline API
- Dynamic tokenization and batching

---

## Installation

Install required libraries:

```bash
pip install transformers datasets torch scikit-learn huggingface_hub
```

---

## Training

Run the training script:

```bash
python finetuning_minilm_for_emotion.py
```

This will:

- Download dataset
- Tokenize text
- Fine-tune MiniLM
- Save model locally
- Push model to Hugging Face Hub

---

---

## Inference Example

```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="nikk118/minilm-finetuned-emotion"
)

pipe("I am very excited about Gen AI")
```

Output:

```
[{'label': 'joy', 'score': 0.99}]
```

---

## Hugging Face Model

https://huggingface.co/nikk118/minilm-finetuned-emotion

---

## Project Structure

```
emotion-classification/
│
├── README.md
├── finetuning_minilm_for_emotion.py
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn

---

## Author

GitHub: https://github.com/Nikk118  
Hugging Face: https://huggingface.co/nikk118

---

## Future Improvements

- Add early stopping
- Hyperparameter tuning
- Deploy using FastAPI
- Convert to ONNX for faster inference
