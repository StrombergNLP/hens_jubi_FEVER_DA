import random
import pandas as pd

# Shuffle the input: This verifies the importance of word (or sentence) order. 
# If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the 
# model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
def shuffle_column(df, column):
     df[column] = df.apply(lambda x: shuffle_input(x[column]), axis=1)
     return df

def shuffle_input(words):
    words = words.split(' ')
    random.shuffle(words)
    words = ' '.join(words)
    return words

# Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? 
# How much with all random labels? If scores don't change much, the model probably didn't learning anything 
# interesting about the task.
def assign_random_labels(df, percent):
    x = int((len(df.claim) * percent) / 100)
    labels = df.label.tolist()
    label_head = labels[:x+1]
    random.shuffle(label_head)
    labels[:x+1] = label_head
    df.label = labels
    return df


# Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are 
# replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, 
# such as sentence length or distribution of function words.

DATA_PATH = '../Classifier/data/annotations-filled-nei.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True)

# ---------
# SHUFFLING 
# ---------
# data_df = shuffle_column(data_df, 'claim')
# data_df.evidence = data_df.evidence.transform(lambda x: ' '.join(x))
# data_df = shuffle_column(data_df, 'evidence')

# -------------
# RANDOM LABELS
# -------------
data_df = assign_random_labels(data_df, 100)

# ------------
# SAVE TO FILE 
# ------------
data_df.to_json('100percent_random_labels.jsonl',orient='records', lines=True)