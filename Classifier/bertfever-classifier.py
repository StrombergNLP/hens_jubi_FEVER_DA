import json
import torch
import transformers
import math
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix

#--------
# Method Definitions
#--------

def read_config():
    """ Take the config path as a string and return the config parameters in a map """
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return (config['data_path'], config['pretrained_model'], config['max_len'], 
            config['batch_size'], config['num_labels'], config['learning_rate'], 
            config['num_epochs'], config['test_size'], bool(config['enable_plotting']))

def drop_duplicate_claims():
    """ Drops rows with duplicate values in claim column. Modifies DF in place! """
    len_with_dupes = len(data_df['claim'])
    data_df.drop_duplicates(subset='claim', inplace=True)
    len_no_dupes = len(data_df['claim'])
    print('Dropped {} duplicate rows.'.format(len_with_dupes - len_no_dupes))

def concatenate_evidence():
    """ Concatenate the evidence for each claim into one string. Edits df in place. """
    data_df.evidence = data_df.evidence.transform(lambda x: ' '.join(x))

def convert_labels():
    """ Convert labels to numeric values. Edits the dataframe in place. """

    labels_vals = {
        'Refuted': 0,
        'Supported': 1, 
        'NotEnoughInfo': 2    
    }

    data_df.label = data_df.label.apply(lambda x: labels_vals[x])

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
    """ 
    Split data inputs to training and validation data. 
    Inputs can be various iterables.
    """
    train_0, valid_0, train_1, valid_1 = train_test_split(input_0, input_1, random_state=1234, test_size=TEST_SIZE) # For testing purpose we are setting a specific random seed
    return train_0, valid_0, train_1, valid_1 

def initialise_dataloader(input_ids, attention_masks, labels, token_type_ids):
    """
    Taking parameters as iterables and return a dataloader.
    """
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    type_tensor = torch.tensor(token_type_ids, dtype=torch.long)

    dataset = TensorDataset(input_tensor, mask_tensor, label_tensor, type_tensor)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader

def initialise_optimiser():
    """ Create optimiser with prespecified hyperparameters and return it. """

    param_optimiser = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     { "params" : [p for n, p in param_optimiser if not any(nd in n for nd in no_decay)], 
    #     "weight_decay_rate" : 0.01 },
    #     { "params" : [p for n, p in param_optimiser if any(nd in n for nd in no_decay)], 
    #     "weight_decay_rate" : 0.0 },
    # ]

    # AdamW used to be BertAdam 
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100

    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = transformers.get_linear_schedule_with_warmup(optimiser, num_warmup_steps, num_total_steps)

    return optimiser, scheduler

def training_epoch():
    """
    Perform one training epoch. Classify training data in batches, optimise model based on the loss.
    """ 
    model.train()

    for batch in train_dataloader:
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        optimiser.zero_grad()       # Clear gradients
        model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)    # Forward pass
        loss = model_output[0]

        loss.backward()     # Backward pass 
        optimiser.step()
        scheduler.step()

def validation_epoch():
    """ Perform one validation epoch. Classify validation data in batches, calculate f1 scores. """
    model.eval()
    for batch in validation_dataloader: 
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        # print('Batch with size: {}'.format(len(batch_labels)))

        with torch.no_grad():       # Not computing gradients, saving memory and time 
            model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            logits = model_output[1]

        evaluate_model(logits, batch_labels)
        
def train_model():

    for epoch in range(NUM_EPOCHS):
        print('... Training epoch {}'.format(epoch+1))
        training_epoch()
        print('... Validating epoch {}'.format(epoch+1))
        validation_epoch()

def evaluate_model(logits, batch_labels):
    """ Use F1-score and confusion matrix to evaluate model performance"""
    preds = np.argmax(logits, axis=1).flatten()     # This gives us the flat predictions

    if(ENABLE_PLOTTING):
        # F1 Score
        micro_f1 = f1_score(batch_labels, preds, average='micro')
        macro_f1 = f1_score(batch_labels, preds, average='macro')
        print('Micro f1: {}'.format(micro_f1))
        print('Macro f1: {}'.format(macro_f1))

        # Confusion matrix
        c_matrix = confusion_matrix(batch_labels.tolist(), preds.tolist())
        print(c_matrix)
        plot_c_matrix(c_matrix)

def plot_c_matrix(c_matrix):
    dimension = int(math.sqrt(c_matrix.size))
    df_cm = pd.DataFrame(c_matrix, range(dimension), range(dimension))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={'size':16})    # font size
    plt.show()

#--------
# Main
#--------

start_time = datetime.now()

print('Reading config...')
CONFIG_PATH = 'config.json'
DATA_PATH, PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE, NUM_LABELS, LEARNING_RATE, NUM_EPOCHS, TEST_SIZE, ENABLE_PLOTTING = read_config()
print('Reading config complete.')

print('Reading data...')
data_df = pd.read_json(DATA_PATH, lines=True)

print('Reading data complete. Loaded {} annotations'.format(len(data_df['claim'])))

print('Pre-processing data...')
drop_duplicate_claims()
concatenate_evidence()
convert_labels()
print('Pre-processing complete.')

print('Preparing data...')
input_ids, token_type_ids = tokenize_inputs()
attention_masks = generate_attention_masks()
train_inputs, validation_inputs, train_labels, validation_labels = split_data(input_ids, data_df.label.tolist())
train_masks, validation_masks, train_token_type_ids, validation_token_type_ids = split_data(attention_masks, token_type_ids)
print('Preparing data complete.')

print('Initialising dataloader...')
train_dataloader = initialise_dataloader(train_inputs, train_masks, train_labels, train_token_type_ids)
validation_dataloader = initialise_dataloader(validation_inputs, validation_masks, validation_labels, validation_token_type_ids)
print('Initialising dataloader complete.')

print('Initialising model...')
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_LABELS)
optimiser, scheduler = initialise_optimiser()
print('Initialising model complete.')

print('Training model...')
train_model()
print('Training model complete.')

print('Execution complete. Execution time: {}.'.format(datetime.now()-start_time))