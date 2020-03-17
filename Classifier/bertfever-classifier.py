import json
import os
import torch
import transformers
import math
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample

#--------
# Method Definitions
#--------

def read_config():
    """ Take the config path as a string and return the config parameters in a map. """
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)
    return (config['data_path'], config['pretrained_model'], config['max_len'], 
            config['batch_size'], config['num_labels'], config['learning_rate'], 
            config['num_epochs'], config['test_size'], bool(config['enable_plotting']), 
            config['output_dir'], bool(config['skip_training']), config['data_sample'], bool(config['enable_cuda']))

def check_output_path():
    """ Check that output path exists, otherwise create it. """
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print('Output directory was not found. Created /{}.'.format(OUTPUT_DIR))

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

def balance_data():
    supported_df = data_df[data_df['label'] == 1]
    refuted_df = data_df[data_df['label'] == 0]
    nei_df = data_df[data_df['label'] == 2]

    major_len = max([len(supported_df.label), len(refuted_df.label), len(nei_df.label)])
    combined_df = pd.DataFrame(columns=['claim', 'entity', 'evidence', 'label'])

    for df in [supported_df, refuted_df, nei_df]:
        df = resample(df, replace=True, n_samples=major_len)
        combined_df = combined_df.append(df)

    return combined_df.sample(frac=1)   # shuffling

def tokenize_inputs():
    """ Return tokenized input ids and token type ids. """
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL, do_lower_case=True)
    input_ids_0 = generate_input_ids(data_df.claim, tokenizer)
    input_ids_1 = generate_input_ids(data_df.evidence, tokenizer)
    token_type_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.create_token_type_ids_from_sequences)
    input_ids = transform_input_ids(input_ids_0, input_ids_1, tokenizer.build_inputs_with_special_tokens) # Add special tokens like [CLS] and [SEP]
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

    input_tensor = create_padded_tensor(input_ids)
    mask_tensor = create_padded_tensor(attention_masks)
    type_tensor = create_padded_tensor(token_type_ids)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if ENABLE_CUDA: label_tensor = label_tensor.cuda()

    dataset = TensorDataset(input_tensor, mask_tensor, label_tensor, type_tensor)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader

def initialise_model():
    """ Initialise model and class weights criterion. Move to GRPU if CUDA is enabled. Return model and criterion. """
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_LABELS)
    class_weights = torch.FloatTensor([1.21, 1.0, 2.68])
    if ENABLE_CUDA:  class_weights = class_weights.cuda()
    criterion = CrossEntropyLoss(weight=class_weights)
    if ENABLE_CUDA: model = model.cuda()
    return model, criterion

def create_padded_tensor(sequence):
    """
    Take a list of variable length lists.
    Truncate and pad them all to MAX_LEN.
    Return a tensor of dimensions len(sequence) * MAX_LEN.
    """

    sequence = [x[:MAX_LEN] for x in sequence]    # Truncate to MAX_LEN
    
    first_padding = MAX_LEN - len(sequence[0])
    sequence[0] = sequence[0] + [0] * first_padding   # Pad first entry to MAX_LEN

    sequence = [torch.tensor(x) for x in sequence]     # Turn list into tensors
    sequence = pad_sequence(sequence, batch_first=True)     # Pad all lists of tensors to length of the largest (MAX_LEN)

    if ENABLE_CUDA: sequence = sequence.cuda()

    return sequence

def initialise_optimiser():
    """ Create optimiser with prespecified hyperparameters and return it. """

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
    total_loss, training_steps = 0, 0
    for index, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        print('Training batch {} with size {}.'.format(index, batch_labels.size()[0]))
        optimiser.zero_grad()       # Clear gradients
        model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)    # Forward pass
        logits = model_output[1]
        loss = criterion(logits, batch_labels)

        loss.backward()     # Backward pass 
        optimiser.step()
        scheduler.step()

        total_loss += loss.item()
        training_steps += 1
        train_loss.append(loss.item())

    print("Train loss: {}".format(total_loss / training_steps))
    # return loss.item()

def validation_epoch():
    """ Perform one validation epoch. Classify validation data in batches, calculate f1 scores. """
    model.eval()

    epoch_labels = torch.LongTensor()
    epoch_logits = torch.FloatTensor()

    for index, batch in enumerate(validation_dataloader): 
        batch_input_ids, batch_attention_masks, batch_labels, batch_token_type_ids = batch      # Unpack input from dataloader
        print('Validation batch {} with size {}.'.format(index, batch_labels.size()[0]))

        with torch.no_grad():       # Not computing gradients, saving memory and time 
            model_output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            logits = model_output[1]

            # Save batch data to be able to evaluate epoch as a whole
            epoch_labels = torch.cat((epoch_labels, batch_labels.cpu()))
            epoch_logits = torch.cat((epoch_logits, logits.cpu()))

    micro_f1, macro_f1, c_matrix = evaluate_model(epoch_logits, epoch_labels)
    
    return micro_f1, macro_f1, c_matrix
        
def train_model():

    for epoch in trange(NUM_EPOCHS, desc="Epoch"):
        if not SKIP_TRAINING: training_epoch()
        micro_f1, macro_f1, c_matrix = validation_epoch()
    
    return micro_f1, macro_f1, c_matrix

def evaluate_model(logits, epoch_labels):
    """ Use F1-score and confusion matrix to evaluate model performance. Logits and label tensors need to be on CPU. """
    preds = np.argmax(logits, axis=1).flatten()     # This gives us the flat predictions
    
    # F1 Score
    micro_f1 = f1_score(epoch_labels, preds, average='micro')
    macro_f1 = f1_score(epoch_labels, preds, average='macro')

    # Confusion matrix
    c_matrix = confusion_matrix(epoch_labels.tolist(), preds.tolist(), labels=[0, 1, 2])
    
    if(ENABLE_PLOTTING): plot_confusion_matrix(c_matrix)

    print('Micro f1: {}'.format(micro_f1))
    print('Macro f1: {}'.format(macro_f1))
    print(c_matrix)

    return micro_f1, macro_f1, c_matrix

def plot_confusion_matrix(c_matrix):
    df_cm = pd.DataFrame(c_matrix, range(NUM_LABELS), range(NUM_LABELS))
    df_cm.columns = ['R', 'S', 'NEI']
    df_cm['Labels'] = ['R', 'S', 'NEI']
    df_cm = df_cm.set_index('Labels')
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={'size':10})    # font size
    plt.show()

def plot_loss(train_loss):
    """ Plot loss after training the model"""
    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss)
    plt.show()

def export_results():
    results = {
        'config': {
            'data_path': DATA_PATH, 
            'data_sample': DATA_SAMPLE,
            'pretrained_model': PRETRAINED_MODEL, 
            'max_len': MAX_LEN, 
            'batch_size': BATCH_SIZE, 
            'num_labels': NUM_LABELS, 
            'learning_rate': LEARNING_RATE, 
            'skip_training': SKIP_TRAINING,
            'num_epochs': NUM_EPOCHS, 
            'test_size' : TEST_SIZE,
            'enable_cuda': ENABLE_CUDA
        }, 
        'loss': train_loss, 
        'micro_f1': micro_f1, 
        'macro_f1': macro_f1, 
        'confusion_matrix': c_matrix.tolist()
    }
    filename = "{}.json".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    output_path = OUTPUT_DIR + filename 
    with open(output_path, mode='w') as outfile:
        json.dump(results, outfile)

#--------
# Main
#--------

start_time = datetime.now()

print('Reading config...')
CONFIG_PATH = 'config.json'
DATA_PATH, PRETRAINED_MODEL, MAX_LEN, BATCH_SIZE, NUM_LABELS, LEARNING_RATE, NUM_EPOCHS, TEST_SIZE, ENABLE_PLOTTING, OUTPUT_DIR, SKIP_TRAINING, DATA_SAMPLE, ENABLE_CUDA = read_config()
check_output_path()
if ENABLE_CUDA: print('Using CUDA on {}.'.format(torch.cuda.get_device_name(0)))
print('Reading config complete.')

print('Reading data...')
data_df = pd.read_json(DATA_PATH, lines=True)
if DATA_SAMPLE > 0: data_df = data_df.head(DATA_SAMPLE)     # For sampling consistently 
print('Supported count: {}'.format(len(data_df.query("label == 'Supported'"))))
print('Refuted count: {}'.format(len(data_df.query("label == 'Refuted'"))))
print('NotEnoughInfo count: {}'.format(len(data_df.query("label == 'NotEnoughInfo'"))))
print('Reading data complete. Loaded {} annotations.'.format(len(data_df['claim'])))

print('Pre-processing data...')
drop_duplicate_claims()
concatenate_evidence()
convert_labels()
print('Pre-processing complete.')

print('Oversampling data...')
# data_df = balance_data()
# print('Supported count: {}'.format(len(data_df.query("label == 1"))))
# print('Refuted count: {}'.format(len(data_df.query("label == 0"))))
# print('NotEnoughInfo count: {}'.format(len(data_df.query("label == 2"))))
print('Oversampling data complete. Size of df: {}'.format(len(data_df.label)))

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
model, criterion = initialise_model()
optimiser, scheduler = initialise_optimiser()
print('Initialising model complete.')

print('Training model...')
train_loss = []
micro_f1, macro_f1, c_matrix = train_model()
print('Training model complete.')

print('Execution time: {}.'.format(datetime.now()-start_time))

print('Plotting loss...')
plot_loss(train_loss)
print('Plotting loss complete.')

print('Exporting results to json...')
export_results()
print('Export complete.')