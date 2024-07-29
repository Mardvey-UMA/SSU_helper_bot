import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import pandas as pd
import model_path
from sklearn.utils import shuffle
import data_ready
from data_ready import LabelDictInv
import database
from datetime import datetime
import sqlite3
DATABASE_NAME = 'questions.db'

def mark_as_used():
    conn = sqlite3.connect("/home/dev-bot/ssu_project/scripts/" + DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE questions SET used = 1 WHERE used = 0
    ''')
    conn.commit()
    conn.close()

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
        label = int(self.labels[idx])
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

import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_test_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data = data.dropna()
    data['label'] = data['label'].replace(LabelDictInv)
    data['text'].apply(preprocess_text)
    data['label'] = data['label'].astype(int)
    data = data.drop(0).reset_index(drop=True)
    data = shuffle(data)  # Перемешиваем данные
    return list(data.itertuples(index=False, name=None))

def evaluate_model(model, tokenizer, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=8)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch['label'].cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def test_new_model():
    new_model_path = model_path.model_path + '_new'
    tokenizer = BertTokenizer.from_pretrained(new_model_path)
    model = BertForSequenceClassification.from_pretrained(new_model_path)

    test_data = load_test_data('/home/dev-bot/ssu_project/scripts/dataset_test.tsv')
    test_texts, test_labels = zip(*test_data)

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

    accuracy = evaluate_model(model, tokenizer, test_dataset)
    print("Точность новой модели:", accuracy)
    
    best_accuracy = database.get_best_accuracy()
    
    print("Лучшая точность:", best_accuracy)
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Текущая дата и время:", formatted_now)
    
    if accuracy > best_accuracy:
        model.save_pretrained(model_path.model_path)
        tokenizer.save_pretrained(model_path.model_path)
        print("Новая модель успешно сохранена.")
        mark_as_used()  # ТОЛЬКО ЕСЛИ НОВАЯ МОДЕЛЬ ЛУЧШЕ ПРЕДЫДУЩЕЙ
        database.add_model_accuracy(accuracy)  # Записываем точность новой модели в БД
    else:
        print("Новая модель не прошла тестирование и не будет сохранена.")

if __name__ == "__main__":
    test_new_model()
