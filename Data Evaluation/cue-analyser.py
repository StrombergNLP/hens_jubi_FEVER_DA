import pandas as pd
from sklearn.utils import resample

# Load data
# DATA_PATH = '../Data Generation/CommonData/annotations/annotations_00.jsonl'
DATA_PATH = 'FEVER_train.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True).head(1800)
claims = data_df.claim.tolist()
claims = [c[:-1] for c in claims]       # Cut off the final period
print('{} claims loaded.'.format(len(claims)))

def drop_duplicate_claims():
    """ Drops rows with duplicate values in claim column. Modifies DF in place! """
    len_with_dupes = len(data_df['claim'])
    data_df.drop_duplicates(subset='claim', inplace=True)
    len_no_dupes = len(data_df['claim'])
    print('Dropped {} duplicate rows.'.format(len_with_dupes - len_no_dupes))

drop_duplicate_claims()

# Extract n-grams
def extract_ngrams(claim):
    bigrams = [b for b in zip(claim.split(" ")[:-1], claim.split(" ")[1:])]
    bigrams = [" ".join(t) for t in bigrams]
    unigrams = claim.split(" ")
    return unigrams + bigrams

# print(claims)
data_df['ngrams'] = data_df.apply(lambda x: extract_ngrams(x.claim), axis=1)
print(data_df[['claim', 'label']])

def balance_data():
    # supported_df = data_df[data_df['label'] == 'Supported']
    # refuted_df = data_df[data_df['label'] == 'Refuted']
    # nei_df = data_df[data_df['label'] == 'NotEnoughInfo']
    supported_df = data_df[data_df['label'] == 'SUPPORTS']
    refuted_df = data_df[data_df['label'] == 'REFUTES']
    nei_df = data_df[data_df['label'] == 'NOT ENOUGH INFO']

    # major_len = max([len(supported_df.label), len(refuted_df.label), len(nei_df.label)])
    minor_len = min([len(supported_df.label), len(refuted_df.label), len(nei_df.label)])
    combined_df = pd.DataFrame(columns=['claim', 'entity', 'evidence', 'label', 'ngrams'])

    # for df in [supported_df, refuted_df, nei_df]:
    #     df = resample(df, replace=True, n_samples=major_len)    # Oversample
    #     combined_df = combined_df.append(df)

    for df in [supported_df, refuted_df, nei_df]:
        df = df.sample(minor_len) # Undersampling
        combined_df = combined_df.append(df)

    return combined_df.sample(frac=1)   # Shuffle

all_cues_df = pd.DataFrame(columns=['ngrams', 'label', 'max', 'total', 'productivity', 'coverage'])
for i in range(10):
    print('Step {}'.format(i))
    balanced_df = balance_data()

    # Do all of the mathy part for each n-gram

    # labels = ['Refuted', 'Supported', 'NotEnoughInfo']
    labels = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO']
    cues_df = balanced_df[['ngrams', 'claim', 'label']]
    cues_df = cues_df.explode('ngrams')
    cues_df = cues_df.groupby(['ngrams', 'label']).count().reset_index()
    cues_df = cues_df.pivot(index='ngrams', columns='label', values='claim').reset_index()
    cues_df = cues_df.fillna(value=0)

    # cues_df['total'] = cues_df.apply(lambda x: x.Supported + x.Refuted + x.NotEnoughInfo, axis=1)
    # cues_df['max'] = cues_df.apply(lambda x: max([x.Supported, x.Refuted, x.NotEnoughInfo]), axis=1)
    cues_df['total'] = cues_df.apply(lambda x: x['SUPPORTS'] + x['REFUTES'] + x['NOT ENOUGH INFO'], axis=1)
    cues_df['max'] = cues_df.apply(lambda x: max([x['SUPPORTS'], x['REFUTES'], x['NOT ENOUGH INFO']]), axis=1)

    # print(cues_df.sort_values('total'))

    # Productivity π(k) of a cue is defined as the proportion of claims in which the cue k appears and where the label is equal
    # to the label with which k appears most frequently. I.e. π(k) is the chance of predicting the label of a claim in the dataset by chosing
    # the most common label for k
    cues_df['productivity'] = cues_df.apply(lambda x: x['max'] / float(x.total), axis=1)

    # print(cues_df.sort_values('productivity'))

    # Coverage ξk of a cue as the proportion of claims that contain the cue over the total number of
    # claims: ξk = αk/n
    cues_df['coverage'] = cues_df.apply(lambda x: x.total / len(balanced_df.claim), axis=1)
    # print(cues_df.sort_values('total'))
    all_cues_df = all_cues_df.append(cues_df)

all_cues_df = all_cues_df.groupby('ngrams').mean().reset_index()

# Output to file
# all_cues_df.to_json('out.jsonl', orient='records', lines=True)
all_cues_df.to_csv('cue_analyser_results.csv')
