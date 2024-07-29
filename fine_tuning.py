import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import model_path

DATABASE_NAME = 'questions.db'

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def fetch_training_data():
    conn = sqlite3.connect("/home/dev-bot/ssu_project/scripts/" + DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT question, label FROM questions WHERE used = 0
    ''')
    data = cursor.fetchall()
    conn.close()
    return data

def load_test_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    return list(data.itertuples(index=False, name=None))

def fine_tune_model():
    data = fetch_training_data()

    if len(data) == 0:
        print("Нет новых данных для дообучения.")
        return

    texts, labels = zip(*data)
    labels = list(labels)

    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    y = [label_dict[label] for label in labels]

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    tokenizer = BertTokenizer.from_pretrained(model_path.model_path)
    model = BertForSequenceClassification.from_pretrained(model_path.model_path)

    dataset = CustomDataset(texts, y, tokenizer)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=1e-5,
        lr_scheduler_type='linear'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    new_model_path = model_path.model_path + '_new'
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)
    
if __name__ == "__main__":
    fine_tune_model()
