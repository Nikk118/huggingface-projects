from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
import torch
from torch import nn
from sklearn.metrics import f1_score

# Load dataset
emotion_dataset = load_dataset("emotion")

# label mappings
features = emotion_dataset["train"].features
id2label = {idx: features["label"].int2str(idx) for idx in range(6)}
label2id = {v:k for k,v in id2label.items()}

# Tokenizer
model_ckpt = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

emotion_dataset = emotion_dataset.map(tokenize, batched=True)

# Rename label column
emotion_dataset = emotion_dataset.rename_column("label", "labels")

# Set torch format
emotion_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# Class weights
import pandas as pd
emotion_df = emotion_dataset["train"].to_pandas()

class_weights = (
    1 - emotion_df["labels"].value_counts().sort_index() / len(emotion_df)
).values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Custom trainer
class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.get("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)

# Metrics
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")

    return {"f1": f1}

# Training args
training_args = TrainingArguments(
    output_dir="minilm-finetuned-emotion",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    eval_strategy="epoch",
    fp16=True,
    push_to_hub=True
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=emotion_dataset["train"],
    eval_dataset=emotion_dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.push_to_hub()

# Inference
pipe = pipeline(
    "text-classification",
    model="nikk118/minilm-finetuned-emotion"
)

pipe("i am really excited about gen ai")
