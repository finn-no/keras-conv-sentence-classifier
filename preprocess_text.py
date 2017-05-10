import nltk
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

def tokenize_a_doc(s):
    try:
        tok = word_tokenize(' '.join(s.split(" ")).lower())
    except:
        tok = ""
    return tok

def vectorize_text(wordlist,wdic):
    return [wdic.get(word) if wdic.get(word) is not None else 0 for word in wordlist]
    
'''
# Arguments
    x: numpy array of text data that wants to be modelled
    name: just a fancy name used to save word2vec embeddings etc
    word2vec: If None, new word2vec will be generated when calling word2vec_init.
        If a path, it will try to load existing a word2vec file (nb: custom format)
    word_dim: Dimensionality of word2vec vectors if new ones are trained.
'''

class Preprocess_text:
    def __init__(self,x,name="textdata",word2vec_path=None, word_dim=50):
        # Download nltk data if it doesnt exist
        nltk.download('punkt')
        self.name = name
        self.df = pd.DataFrame(x,columns=["text"])
        self.word2vec_path = word2vec_path
        self.word_dim = word_dim
        
    def tokenize_text(self):
        self.df['tokenized'] = self.df.text.map(tokenize_a_doc)
        
    def word2vec_init(self,word_dim = 50):
        if self.word2vec_path is None:
            # Create word2vecs from text data
            sentences = self.df.tokenized.values.tolist()
            print 'Training on %d documents.' %(len(sentences))
            
            # Build word2vec model
            gensimmodel = gensim.models.word2vec.Word2Vec(sentences, size=self.word_dim, window=5, min_count=5, workers=15)
            print 'w2v model generated.'

            # Go from gensim to numpy arrays.
            # Add first index to be zero index for missing words.
            words = ["UNK"]
            words.extend(gensimmodel.wv.vocab.keys())

            vecs0 = np.repeat(0,self.word_dim)
            vecs = np.vstack(map(lambda w: gensimmodel[w], gensimmodel.wv.vocab.keys()))
            self.vectors = np.vstack([vecs0,vecs])
            self.words = words
            np.savez("word2vec"+self.name+".npz",vectors = self.vectors, words = self.words)
            print 'w2v saved and initialized.'
        else:
            w2v = np.load(self.word2vec_path)
            self.vectors = w2v['vectors']
            self.words = w2v['words']
            
    def vectorize_text(self):
        words = self.words
        wdic = {word: i for i, word in enumerate(words)} 
        self.df['vectorized'] = self.df.tokenized.map(lambda x: vectorize_text(x,wdic))
        
    def run_all(self):
        self.tokenize_text()
        self.word2vec_init()
        self.vectorize_text()