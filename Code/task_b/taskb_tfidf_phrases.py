from datasets import load_dataset
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import nltk
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords

dataset = load_dataset("PiC/phrase_similarity")

data = pd.DataFrame(dataset['test'])

drop=['sentence1','sentence2','idx']
data.drop(columns=drop, inplace=True)

# file_path = 'dataset_test.txt'

table = PrettyTable()
table.field_names = data.columns
for index, row in data.iterrows():
    table.add_row(row)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    text = text.lower()

    text = ''.join(char for char in text if char not in punctuation)

    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

data['phrase1'] = data['phrase1'].apply(clean_text)
data['phrase2'] = data['phrase2'].apply(clean_text)

# with open(file_path, "w") as file:
    # file.write(str(table))
    
word_frequency = {}

for sent in data['phrase1']:
    words = sent.split(" ")
    for word in words:
        if word not in word_frequency:
            word_frequency[word] = 0

for sent in data['phrase2']:
    words = sent.split(" ")
    for word in words:
        if word not in word_frequency:
            word_frequency[word] = 0

# print(word_frequency)
            
# calculating idf
idf_words=word_frequency.copy()
total_docs=len(data['phrase1'])*2

for index, row in data.iterrows():
    temp_freq=word_frequency.copy()
    words = row['phrase1'].split(" ")
    for word in words:
        if temp_freq[word]==0:
            temp_freq[word]=1
    
    for word,frequency in temp_freq.items():
        if frequency==1:
            idf_words[word]+=1
    
    temp_freq_2=word_frequency.copy()
    words=row['phrase2'].split(" ")
    for word in words:
        if temp_freq_2[word]==0:
            temp_freq_2[word]=1
    
    for word,frequency in temp_freq_2.items():
        if frequency==1:
            idf_words[word]+=1

            
output_table = PrettyTable(['Word'] + list(word_frequency.keys()))

# calculating tf
sentence_word_frequencies = []

for index, row in data.iterrows():
    sentence_frequency_1 = word_frequency.copy()
    sentence_frequency_2=word_frequency.copy()
    
    words = row['phrase1'].split(" ")
    for word in words:
        sentence_frequency_1[word] += 1
    
    words = row['phrase2'].split(" ")
    for word in words:
        sentence_frequency_2[word] += 1
    
    for word,frequency in sentence_frequency_1.items():
        temp_idf=math.log(total_docs/idf_words[word])
        sentence_frequency_1[word]=sentence_frequency_1[word]*temp_idf
    
    for word,frequency in sentence_frequency_2.items():
        temp_idf=math.log(total_docs/idf_words[word])
        sentence_frequency_2[word]=sentence_frequency_2[word]*temp_idf
    
    # sentence_word_frequencies.append(sentence_frequency)
    output_table.add_row([index*2] + list(sentence_frequency_1.values()))
    output_table.add_row([index*2+1] + list(sentence_frequency_2.values()))

# output_file_path = 'word_frequencies.txt'
# with open(output_file_path, "w") as file:
    # file.write(str(output_table))

accuracy=[]

for i in range(0, len(output_table._rows),2):
    dot_product=sum(a*b for a,b in zip(output_table._rows[i],output_table._rows[i+1]))
    magnitude_1 = math.sqrt(sum(a**2 for a in output_table._rows[i]))
    magnitude_2 = math.sqrt(sum(b**2 for b in output_table._rows[i+1]))
    score=(round((dot_product / (magnitude_1 * magnitude_2)) * 100, 2))
    if score > 95:
        accuracy.append(1)
    else:
        accuracy.append(0)


table.add_column("Similarity",accuracy)
file_path_output="phrases_output_tfidf.txt"

with open(file_path_output, "w") as file:
    file.write(str(table))

data['Similarity'] = accuracy
predicted_labels = data['label']
actual_labels = data['Similarity']
accuracy_percentage = np.sum(actual_labels == predicted_labels) / len(actual_labels) * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")
