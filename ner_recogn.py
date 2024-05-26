import spacy
from spacy import displacy
import re

nlp = spacy.load("ru_core_news_md")

def extract_entities(text):
    text = text.split()
    text = [word.capitalize() for word in text]
    text = ' '.join(text)
    
    doc = nlp(text)

    surnames = [ent.text for ent in doc.ents if ent.label_ == 'PER']
    group_numbers = re.findall(r'\b\d{3}\b', text)
    
    return [surnames, group_numbers]
