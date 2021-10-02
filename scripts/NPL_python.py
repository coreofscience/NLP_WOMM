import gensim 
import pandas as pd
import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
import spacy
import logging
import warnings
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('stopwords')

# Enable logging para gensim - optional (para generar mensajes en forma de logs)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# df3 = pd.read_csv("data/wos_txt.csv")

df3 = r.df  # Call from python

df3.shape

# Convierte a lista
data = df3.values.astype(str).tolist()

## Tokenización a nivel de palabra y limpieza de texto


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words)

# Creación de bigramas y trigramas

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=1, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1]]])

# Eliminación de Stop Words y lematizado

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Eliminación de Stop Words
data_words_nostops = remove_stopwords(data_words)

# Determinación de bigramas
data_words_bigrams = make_bigrams(data_words_nostops)

# Modelo de spacy en inglés
nlp = spacy.load('en', disable=['parser', 'ner'])

# Lematización manteniendo sustantivos, adverbios, adejtivos y verbos
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:10])

