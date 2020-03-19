import pandas as pd

# Load data
DATA_PATH = '../Data Generation/CommonData/annotations/annotations_00.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True).head(50)
claims = data_df.claim.tolist()
claims = [c[:-1] for c in claims]       # Cut off the final period
print('{} claims loaded.'.format(len(claims)))

# Extract n-grams
def extract_bigrams(claim):
    bigrams = [b for b in zip(claim.split(" ")[:-1], claim.split(" ")[1:])]
    bigrams = [" ".join(t) for t in bigrams]
    return bigrams

def extract_unigrams(claim):
    unigrams = claim.split(" ")
    return unigrams

# print(claims)
data_df['bigrams'] = data_df.apply(lambda x: extract_bigrams(x.claim), axis=1)
data_df['unigrams'] = data_df.apply(lambda x: extract_unigrams(x.claim), axis=1)
# print(data_df)

# Do all of the mathy part for each n-gram

data_df.entity = data_df.entity.apply(lambda x: x[0])
entities = data_df.entity.drop_duplicates()

# Construct T_j(i): ngram_set for each label(j) and entity(i)
ngram_sets = pd.DataFrame(columns=['entity', 'label', 'ngrams'])

for e in entities:
    ngrams = {'Refuted': [], 'Supported': [], 'NotEnoughInfo': []}
    entity_rows = data_df[data_df.entity == e]
    # print(entity_rows)
    for index, row in entity_rows.iterrows():
        ngrams[row.label] = ngrams[row.label] + row.unigrams + row.bigrams
    for k in ngrams.keys():
        ngram_sets = ngram_sets.append({'entity': e, 'label': k, 'ngrams': ngrams[k]}, ignore_index=True)

ngram_sets.ngrams = ngram_sets.ngrams.apply(lambda x: set(x))
# print(ngram_sets)

ngram_sets.to_json('out.jsonl', orient='records', lines=True)

# Get a set of cues
cues = set()
ngram_sets.ngrams.apply(lambda x: cues.update(x))
cues_df = pd.DataFrame(cues, columns=['cue'])

# print(cues)

# applicability αk a cue’s applicability as the number of data points (i.e. entities)
# where it occurs with one label but not the other

LABELS = ['Refuted', 'Supported', 'NotEnoughInfo']

def calculate_applicability(row):
    k = row.cue
    print('Cue: ' + k)
    applicability = 0
    for e in entities:
        count = 0
        for l in LABELS:
            row = ngram_sets.query("entity == \"{}\" and label == '{}'".format(e, l))
            # print(row.ngrams.values[0])
            if k in row.ngrams.values[0]:
                count += 1
        
        if count == 1: applicability += 1
    return applicability

cues_df['applicability'] = cues_df.apply(lambda x: calculate_applicability(x), axis=1)

# Productivity πk of a cue is defined as the proportion of applicable data points for
# which it predicts the correct answer


# Coverage ξk of a cue as the proportion of applicable cases over the total number of
# data points: ξk = αk/n

def calculate_coverage(row):
    a = row.applicability
    coverage = a / float(len(entities))
    return coverage

cues_df['coverage'] = cues_df.apply(lambda x: calculate_coverage(x), axis=1)

# Save results
cues_df.to_csv('out.csv')
print(cues_df)