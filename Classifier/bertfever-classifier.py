import pandas as pd
import json
from datetime import datetime

# Method definitions
def read_config():
    """ Take the config path as a string and return the config parameters in a map """
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return (config['data_path'], config['pretrained_model'], config['max_len'], 
            config['batch_size'], config['num_labels'], config['learning_rate'], 
            config['num_epochs'], config['test_size'], config['wiki_path'])

def drop_duplicate_claims():
    """ Drops rows with duplicate values in claim column. Modifies DF in place! """
    len_with_dupes = len(data_df['claim'])
    data_df.drop_duplicates(subset='claim', inplace=True)
    len_no_dupes = len(data_df['claim'])
    print('Dropped {} duplicate rows.'.format(len_with_dupes - len_no_dupes))

def add_nei_evidence():
    """ Add the entity's abstract as evidence to each claim with NEI label. Modifies DF in place! """
    for index, row in data_df.iterrows():
        if row.label == 'NotEnoughInfo':
            row.evidence = lookup_abstract(row.entity[0])

def lookup_abstract(entity):
    """ Lookup entity in wiki dataframe and return its abstract in a list """
    row = wiki_df[wiki_df.Title == entity]
    abstract = row.Abstract.values[0]
    return [abstract]

# Main
start_time = datetime.now()

print('Reading config...')
CONFIG_PATH = 'config.json'
DATA_PATH, PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE, NUM_LABELS, LEARNING_RATE, NUM_EPOCHS, TEST_SIZE, WIKI_PATH = read_config()
print('Reading config complete.')

print('Reading data...')
data_df = pd.read_json(DATA_PATH, lines=True)
wiki_df = pd.read_json(WIKI_PATH, lines=True)
print('Reading data complete. Loaded {} annotations and {} wiki articles.'.format(len(data_df['claim']), len(wiki_df['Title'])))

print('Pre-processing data...')
drop_duplicate_claims()
add_nei_evidence()
print('Pre-processing complete.')

print(data_df[['evidence', 'label']])

