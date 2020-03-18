# Following https://www.machinelearningplus.com/nlp/cosine-similarity/

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import numpy as np

def calculate_cos_similarity(row):
    sparse_matrix = vectorizer.fit_transform([row.claim, row.evidence])
    dense_matrix = sparse_matrix.todense()
    matrix_df = pd.DataFrame(dense_matrix, columns=vectorizer.get_feature_names(), index=['claim', 'evidence'])
    return cosine_similarity(matrix_df)[0][1]    

# Read data
data_df = pd.read_json('data/annotations-filled-nei.jsonl', lines=True)
data_df.evidence = data_df.evidence.apply(lambda x: ' '.join(x))    # Concat evidence for each row into one string

vectorizer = TfidfVectorizer() 

data_df['cosine_similarity'] = data_df.apply(lambda x: calculate_cos_similarity(x), axis=1)
print(data_df)

# Split
TEST_SIZE = 0.1
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE)

LABELS = ['Refuted', 'Supported', 'NotEnoughInfo']
means = []

# 'Training'
for label in LABELS:
    mean = train_df[train_df.label == label].cosine_similarity.mean()
    print('Mean cosine similarity for {}: {}'.format(label, mean))
    means.append(mean)

# 'Testing', i.e. calculating distance from consine similarity of each claim-evidence pair to the means of the labels
def find_nearest_label(row):
    distances = [abs(row.cosine_similarity - m) for m in means]    # Calculate distance to each label's mean
    prediction = LABELS[np.argmin(distances)]       # Get label with the smallest distance
    return prediction

test_df['prediction'] = test_df.apply(lambda x: find_nearest_label(x), axis=1)

print(test_df)

# Convert labels to numbers
labels_vals = {
    'Refuted': 0,
    'Supported': 1, 
    'NotEnoughInfo': 2    
}
test_df.label = test_df.label.apply(lambda x: labels_vals[x])
test_df.prediction = test_df.prediction.apply(lambda x: labels_vals[x])

# Evaluate  
labels = test_df.label
preds = test_df.prediction

# F1 Score
micro_f1 = f1_score(labels, preds, average='micro')
macro_f1 = f1_score(labels, preds, average='macro')

# Confusion matrix
c_matrix = confusion_matrix(labels.tolist(), preds.tolist(), labels=[0, 1, 2])

print('Micro f1: {}'.format(micro_f1))
print('Macro f1: {}'.format(macro_f1))
print(c_matrix)
