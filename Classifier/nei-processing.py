import pandas as pd

def add_nei_evidence():
    """ Add the entity's abstract as evidence to each claim with NEI label. Modifies DF in place! """
    for index, row in data_df.iterrows():
        if row.label == 'NotEnoughInfo':
            row.evidence = lookup_abstract(row.entity[0])

def lookup_abstract(entity):
    """ Lookup entity in wiki dataframe and return its abstract in a list """
    row = wiki_df[wiki_df.Title == entity]
    abstract = row.Abstract.values[0] if len(row.Abstract.values) else ''
    return [abstract]

print('Reading data...')
DATA_PATH = "data/annotations.jsonl"
WIKI_PATH = "data/dawiki-latest-pages-articles-parsed.jsonl"
data_df = pd.read_json(DATA_PATH, lines=True)
wiki_df = pd.read_json(WIKI_PATH, lines=True)
print('Reading data complete.')

print('Adding evidence to NEI-labbelled claims...')
add_nei_evidence()
print('Adding evidence complete.')

data_df.to_json('data/annotations-filled-nei.jsonl', orient='records', lines=True)
print('Saved {} claims to file.'.format(len(data_df['claim'])))