from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

words = None


def one_hot_words(values):
    words, unique_inverse = np.unique(values, return_inverse=True)
    return to_categorical(unique_inverse)


def get_padded_input(values, max_doc_length=15):
    return pad_sequences(values, maxlen=max_doc_length, padding='post', truncating='post')


def load_finn_torget_embeddings(embedding_path):
    return np.load(embedding_path)['vectors']

def tokenize_a_doc(doc, embedding_words, num_words=25):
    try:
        tokenized_doc = word_tokenize(' '.join(doc.split(" ")[:num_words]).lower())
        return indexize_text(tokenized_doc, embedding_words)
    except:
        return [0]


def indexize_text(doc, embedding_words):
    vect = []
    for word in doc:
        word_idx = embedding_words.get(word)
        if word_idx is not None:
            vect.append(word_idx)
        else:
            vect.append(0)
    return vect