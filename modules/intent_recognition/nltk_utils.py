import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from pymystem3 import Mystem
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

mystem = Mystem()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("russian"))


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokens, words):
    # stem each word
    tokens = [stem(word) for word in tokens]
    tokens = [token for token in tokens if token.strip() not in punctuation]
    tokens = [token for token in tokens if not token.lower() in stop_words]

    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokens:
            bag[idx] = 1

    return bag