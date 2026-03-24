import pandas as pd
import os
import glob
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil

# ==========================================
# 1. LOADING DIVERSE DATASETS (BENIGN & MALICIOUS)
# ==========================================
print("Loading datasets (HuggingFace & Kaggle)...")

# --- MALICIOUS DATASETS (Label 1) ---
ds_dataset = load_dataset("deepset/prompt-injections")
df_deepset = pd.DataFrame(ds_dataset['train'])

kaggle_dfs = []
for file in glob.glob("/content/*.csv"):
    try:
        temp_df = pd.read_csv(file)
        if 'Prompt' in temp_df.columns:
            temp_df.rename(columns={'Prompt': 'text'}, inplace=True)
            temp_df['label'] = 1
            kaggle_dfs.append(temp_df[['text', 'label']])
    except Exception as e:
        print(f"Skipping {file}: {e}")
df_kaggle = pd.concat(kaggle_dfs, ignore_index=True) if kaggle_dfs else pd.DataFrame(columns=['text', 'label'])

df_malicious = pd.concat([df_deepset, df_kaggle], ignore_index=True)
df_malicious.dropna(subset=['text', 'label'], inplace=True)
df_malicious['label'] = 1

# --- BENIGN DATASETS (Label 0) ---
print("Downloading Alpaca cleaned dataset for general benign baseline...")
alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
df_alpaca = pd.DataFrame(alpaca_dataset)
df_alpaca.rename(columns={'instruction': 'text'}, inplace=True)

print("Downloading CodeAlpaca dataset for complex technical/engineering baseline...")
code_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
df_code = pd.DataFrame(code_dataset)
df_code.rename(columns={'instruction': 'text'}, inplace=True)

df_benign = pd.concat([df_alpaca, df_code], ignore_index=True)
df_benign.dropna(subset=['text'], inplace=True)
df_benign['label'] = 0

# ==========================================
# 2. COMBINING ALL DATA
# ==========================================
print(f"Total Malicious Samples: {len(df_malicious)}")
print(f"Total Benign Samples (Highly Diverse): {len(df_benign)}")
print("Merging datasets to create a robust, context-aware AI...")

df_final = pd.concat([df_benign, df_malicious], ignore_index=True)
df_final.drop_duplicates(subset=['text'], inplace=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle perfectly

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_final['text'].tolist(), df_final['label'].tolist(), test_size=0.15, random_state=42
)

# ==========================================
# 3. TOKENIZATION AND DATASET SETUP
# ==========================================
print("Configuring DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class PromptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_dataset = PromptDataset(train_encodings, train_labels)
test_dataset = PromptDataset(test_encodings, test_labels)

# ==========================================
# 4. TRAINING (CONTEXT-AWARE FINE-TUNING)
# ==========================================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results_context_aware',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("\nTraining context-aware model! It will take 15-20 minutes with GPU...")
trainer.train()

# ==========================================
# 5. SAVING AND EVALUATING THE MODEL
# ==========================================
final_dir = "./distilbert_context_aware_model"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print("\n--- OFFICIAL METRICS ---")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
acc = accuracy_score(labels, preds)
print(f"Accuracy : {acc:.4f}\nPrecision: {precision:.4f}\nRecall   : {recall:.4f}\nF1 Score : {f1:.4f}")

print(f"\nTraining complete! Zipping the model...")
shutil.make_archive(final_dir, 'zip', final_dir)
print(f"Done!")