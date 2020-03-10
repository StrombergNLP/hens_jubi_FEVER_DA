import pandas as pd
import json
from datetime import datetime
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#--------
# Method Definitions
#--------

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

def concatenate_evidence():
    """ Concatenate the evidence for each claim into one string. Edits df in place. """
    data_df.evidence = data_df.evidence.transform(lambda x: ' '.join(x))

def tokenize_inputs():
    """ Return tokenized input ids and token type ids. """
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL, do_lower_case=True)
    input_ids_0 = generate_input_ids(data_df.claim, tokenizer)
    input_ids_1 = generate_input_ids(data_df.evidence, tokenizer)
    token_type_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.create_token_type_ids_from_sequences)
    input_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.build_inputs_with_special_tokens) # Add special tokens like [CLS] and [SEP]
    token_type_ids = pad_sequences(token_type_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    return input_ids, token_type_ids

def generate_input_ids(sequences, tokenizer):
    """ Take a list of sequences (claim or evidence) and a tokenizer. Return input ids for sequences as a list."""
    tokenized_text = [tokenizer.tokenize(seq) for seq in sequences] # Borgmester -> Borg - mester
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_text] # Borg -> float
    return input_ids

def transform_input_ids(input_ids_0, input_ids_1, tokenizer_func):
    """
    Take the input ids for sequences 0 and 1 (claim and evidence) and a tokenizer function.
    Apply function to tuples of claim-evidence.
    Return list of token type ids.
    """
    transformed_ids = list(map(
        lambda ids_tuple: tokenizer_func(ids_tuple[0], ids_tuple[1]),
        zip(input_ids_0, input_ids_1)
    ))
    return transformed_ids

def generate_attention_masks():
    """ 
    For every sequence in the input_id, generate attention masks.
    1 for every non-zero id, 0 for padding.
    Return attention masks as list.
    """
    attention_masks = list(map(lambda x: [float(i > 0) for i in x], input_ids))
    return attention_masks

def split_data(input_0, input_1):
    """ Split data inputs to training and validation data. """
    train_0, valid_0, train_1, valid_1 = train_test_split(input_0, input_1, random_state=1234, test_size=TEST_SIZE) # For testing purpose we are setting a specific random seed
    return train_0, valid_0, train_1, valid_1 

#--------
# Main
#--------

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
concatenate_evidence()
print('Pre-processing complete.')

#print(data_df[['evidence', 'label']])

print('Preparing data...')
input_ids, token_type_ids = tokenize_inputs()
attention_masks = generate_attention_masks()
train_inputs, validation_inputs, train_labels, validation_labels = split_data(input_ids, data_df.label)
train_masks, validation_masks, train_token_type_ids, validation_token_type_ids = split_data(attention_masks, token_type_ids)
print('Preparing data complete.')

print('Initialising dataloader...')

print('Initialising dataloader complete.')