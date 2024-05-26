import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import re
import model_path
from model_path import model_path
import numpy

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_own_model(model_path):
    # Загрузка конфигурации модели
    config = BertConfig.from_pretrained(model_path + '\config.json')

    model = BertForSequenceClassification.from_pretrained(
        model_path + '\model.safetensors',
        config=config
    )

    tokenizer = BertTokenizer.from_pretrained(model_path)

    return tokenizer, model

# Перевод модели в режим оценки
#model.eval()


# Функция для предсказания
def predict(text, tokenizer, model):
    text = preprocess_text(text)

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probabilities.numpy()

## Получение вероятностей классов
#probabilities = predict(text, tokenizer, model)

#print("Вероятности классов:", probabilities)

## Получение класса с максимальной вероятностью
# predicted_class = probabilities.argmax(axis=-1)
# print("Предсказанный класс:", predicted_class)