import random
import pandas as pd
import stanza
import ftfy 

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

def replace_content_words(series, column1):

    if series[column1 + '_tags'] != None:
        for noun in series[column1 + '_tags']['nouns']:
            series[column1] = series[column1].replace(noun, random.sample(nouns, 1)[0])

        for verb in series[column1 + '_tags']['verbs']:
            series[column1] = series[column1].replace(verb, random.sample(verbs, 1)[0])
            
    return series[column1]

def tag_string(string):
    if string != '':
        doc = nlp(string)
        local_nouns = set()
        local_verbs = set()

        for sent in doc.sentences: 
            for word in sent.words:
                if word.upos == 'NOUN':
                    clean_word = ftfy.fix_encoding(word.text)
                    local_nouns.add(clean_word)
                elif word.upos == 'VERB':
                    clean_word = ftfy.fix_encoding(word.text)
                    local_verbs.add(clean_word)
        
        nouns.update(local_nouns)
        verbs.update(local_verbs)

        return {'nouns': local_nouns, 'verbs': local_verbs}    
    
DATA_PATH = '../Classifier/data/annotations-filled-nei.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True)

# ---------
# SHUFFLING 
# ---------
# data_df = shuffle_column(data_df, 'claim')
data_df.evidence = data_df.evidence.transform(lambda x: ' '.join(x))
# data_df = shuffle_column(data_df, 'evidence')

# -------------
# RANDOM LABELS
# -------------
# data_df = assign_random_labels(data_df, 100)

# --------------
# RANDOM CONTENT
# --------------
string = 'Hej hvordan går det med hunden i dag?'
stanza.download('da')
nlp = stanza.Pipeline(lang='da', processors='tokenize,mwt,pos')
doc = nlp(string)
nouns = set()
verbs = set()
counter = 0

print('Processing claims...')
data_df['claim_tags'] = data_df.apply((lambda x: tag_string(x.claim)), axis=1)
print('Processing evidence...')
data_df['evidence_tags'] = data_df.apply((lambda x: tag_string(x.evidence)), axis=1)

print('Replacing claim tags...')
data_df['claim'] = data_df.apply((lambda x: replace_content_words(x, 'claim')), axis=1)
print('Replacing evidence tags...')
data_df['evidence'] = data_df.apply((lambda x: replace_content_words(x, 'evidence')), axis=1)

# ------------
# SAVE TO FILE 
# ------------
data_df.to_json('replaced_content_words.jsonl',orient='records', lines=True)