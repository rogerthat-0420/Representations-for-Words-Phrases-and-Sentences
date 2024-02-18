import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import csv
from prettytable import PrettyTable


sentences=brown.sents()

myfile="browncorpus.txt"
with open(myfile, "w", encoding="utf-8") as file:
    for sentence in sentences:
        file.write(" ".join(sentence) + "\n")

with open(myfile, "r", encoding="utf-8") as file:
    corpus_text = file.read()

brown_tokens=word_tokenize(corpus_text.lower())
stop_words=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
brown_tokens = [lemmatizer.lemmatize(word) for word in brown_tokens if word.isalpha() and word not in stop_words]

from gensim.models import Word2Vec
import pandas as pd
import numpy as np

brown_word2vec_model = Word2Vec(sentences=[brown_tokens],min_count=10,window=5,vector_size=200)
brown_word2vec_model.save("word2vec_model")

mymodel=Word2Vec.load("word2vec_model")
test=pd.read_csv("a_simlex_clean.csv")
    
similarity_vector=[]

def get_antonyms(word):
    antonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())

    return antonyms

def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

    return synonyms

for index, row in test.iterrows():
    w1=word_tokenize(row['word1'])
    w2=word_tokenize(row['word2'])
    w1_string = " ".join(w1)
    w2_string = " ".join(w2)

    synonyms_set_w1=get_synonyms(w1_string)
    antonyms_set_w1=get_antonyms(w1_string)

    if w2_string in synonyms_set_w1:
        score=1
    elif w2_string in antonyms_set_w1:
        score=-1
    elif w1_string in mymodel.wv.key_to_index and w2_string in mymodel.wv.key_to_index:
        score = mymodel.wv.similarity(w1_string, w2_string)
        score=round(score,2)
    else:
        score=10000

    similarity_vector.append(score)

test['COMPARE']=similarity_vector

test['Similarity_Label_mymodel'] = np.where(test['COMPARE'] >= 0.5, 'similar', np.where(test['COMPARE'] == 10000, 'Word not found in model', 'not similar'))

test['Similarity_Label_SimLex999'] = np.where(test['SimLex999'] >= 5, 'similar', 'not similar')

column_widths = {'word1': 15, 'word2': 15, 'SimLex999': 5, 'COMPARE': 15, 'Similarity_Label_mymodel': 11,
                 'Similarity_Label_SimLex999':11}
table = PrettyTable()
table.field_names = test.columns
for index, row in test.iterrows():
    table.add_row(row)

# Write the output to the text file
with open("a_word2vec_output.txt", "w") as file:
    file.write(str(table))

predicted_labels = test['Similarity_Label_mymodel']
actual_labels = test['Similarity_Label_SimLex999']
# Calculating accuracy
accuracy = np.sum(actual_labels == predicted_labels) / len(actual_labels) * 100
print(f"Accuracy: {accuracy:.2f}%")











