import pandas as pd

# Load data
DATA_PATH = '../Data Generation/CommonData/annotations/annotations_00.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True).head(100)
claims = data_df.claim.tolist()
claims = [c[:-1] for c in claims]       # Cut off the final period
print('{} claims loaded.'.format(len(claims)))

# Extract n-grams
def extract_ngrams(claim):
    bigrams = [b for b in zip(claim.split(" ")[:-1], claim.split(" ")[1:])]
    bigrams = [" ".join(t) for t in bigrams]
    unigrams = claim.split(" ")
    return unigrams + bigrams

# print(claims)
data_df['ngrams'] = data_df.apply(lambda x: extract_ngrams(x.claim), axis=1)
print(data_df)
# Do all of the mathy part for each n-gram

labels = ['Refuted', 'Supported', 'NotEnoughInfo']
cues_df = data_df[['ngrams', 'claim', 'label']]
cues_df = cues_df.explode('ngrams')
cues_df = cues_df.groupby(['ngrams', 'label']).count().reset_index()
cues_df = cues_df.pivot(index='ngrams', columns='label', values='claim').reset_index()
cues_df = cues_df.fillna(value=0)
cues_df['total'] = cues_df.apply(lambda x: x.Supported + x.Refuted + x.NotEnoughInfo, axis=1)
print(cues_df)

# Productivity π(k) of a cue is defined as the proportion of claims in which the cue k appears and where the label is equal
# to the label with which k appears most frequently. I.e. π(k) is the chance of predicting the label of a claim in the dataset by chosing
# the most common label for k
# There's a formula for this in the notebook

# Not yet redefined 

# applicability αk a cue’s applicability as the number of data points 
# where it occurs with one label but not the other (not needed for comparability, might not be necessary anymore for calculation either)

# Coverage ξk of a cue as the proportion of applicable cases over the total number of
# data points (claims): ξk = αk/n
