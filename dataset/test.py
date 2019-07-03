from collections import defaultdict
import os

t_path = "./bbc/"
all_docs = defaultdict(lambda: list())

topic_list = list()
text_list = list()

for topic in os.listdir(t_path):
    d_path = t_path + topic + '/'
    i=1
    for f in os.listdir(d_path):
        
        f_path = d_path + f
        file = open(f_path,'r')
        text_list.append(file.read())
        file.close()
        topic_list.append(topic)
        if i>3:
            break
        else:
            continue

from sklearn.model_selection import train_test_split

title_train, title_test, category_train, category_test = train_test_split(text_list, topic_list)
title_train, title_dev, category_train, category_dev = train_test_split(title_train, category_train)

# tokenizing words, romoving stopwords and converting the raw data into numneric data vector
#space model using tfidfvectorizer and tokenizer as follow:

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenizer,
                            stop_words=stop_words)
vectorizer.fit(iter(title_train))
Xtr = vectorizer.transform(iter(title_train))
Xde = vectorizer.transform(iter(title_dev))
Xte = vectorizer.transform(iter(title_test))


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(category_train)
Ytr = encoder.transform(category_train)
Yde = encoder.transform(category_dev)
Yte = encoder.transform(category_test)

print(Xte)
