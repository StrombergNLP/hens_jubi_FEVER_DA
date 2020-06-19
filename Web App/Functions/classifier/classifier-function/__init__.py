import logging

import azure.functions as func
import json
import os
import pathlib
import torch
import transformers
import math
import pandas as pd
import numpy as np
import glob
import ftfy
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

#--------
# Method Definitions
#--------

def read_config(path):
    """ Take the config path as a string and return the config parameters in a map. """
    try:
        path = pathlib.Path(__file__).parent.parent / path
        logging.info('Looking for config at {}.'.format(path))
        with open(path, 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError as e:
        logging.error("Config file not found! Using default values.")
        config = {
            "max_len": 125,
            "batch_size": 16,
            "num_labels": 3,
            "pretrained_model": "Config/models/20e-b32/",
            "enable_cuda": 0
        }
    return config

def convert_labels(df, columns=['label'], reverse=False):
    """ Convert labels in provided columns to numeric values (or back to text). Edits the dataframe in place. """
    if reverse:
        labels_vals = {
            0: 'Refuted',
            1: 'Supported',
            2: 'NotEnoughInfo'
        }
    else:
        labels_vals = {
            'Refuted': 0,
            'Supported': 1, 
            'NotEnoughInfo': 2    
        }

    for col in columns:
        df[col] = df[col].apply(lambda x: labels_vals[x])
    return df

def tokenize_inputs(claim, evidence, config):
    """ Return tokenized input ids and token type ids. """
    model_path = str(pathlib.Path(__file__).parent.parent / config['pretrained_model'])
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    input_ids_0 = generate_input_ids(claim, tokenizer)
    input_ids_1 = generate_input_ids(evidence, tokenizer)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(input_ids_0, input_ids_1)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids_0, input_ids_1)
    return input_ids, token_type_ids

def generate_input_ids(sequence, tokenizer):
    tokenized_text = tokenizer.tokenize(sequence) # Borgmester -> Borg - mester
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_text] # Borg -> float
    return input_ids

def initialise_model(config):
    """ Initialise model. Move to GPU if CUDA is enabled. Return model. """
    model_path = str(pathlib.Path(__file__).parent.parent / config['pretrained_model'])
    logging.info('Looking for model at {}.'.format(model_path))
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=config['num_labels'])
    if bool(config['enable_cuda']): model = model.cuda()
    return model

def create_padded_tensor(sequence, config):
    """
    Take a list of variable length lists.
    Truncate and pad them all to MAX_LEN.
    Return a tensor of dimensions len(sequence) * MAX_LEN.
    """
    max_len = config['max_len']
    sequence = sequence[:max_len]   # Truncate to MAX_LEN
    sequence = torch.tensor([sequence])

    if bool(config['enable_cuda']): sequence = sequence.cuda()

    return sequence

def get_flat_predictions(logits):
    preds = np.argmax(logits, axis=1).flatten()     # This gives us the flat predictions
    return preds


def test_model(model, input_ids, token_type_ids, attention_mask):
    """ Perform the test. Classify test data in batches, calculate f1 scores. """

    with torch.no_grad():       # Not computing gradients, saving memory and time 
        model_output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = model_output[0]

    preds = get_flat_predictions(logits)
    
    return preds


def predict(claim, evidence):
    start_time = datetime.now()

    logging.info('Reading config...')
    CONFIG_PATH = 'classifier-function/classifier-config.json'
    config = read_config(CONFIG_PATH)
    if bool(config['enable_cuda']): logging.info('Using CUDA on {}.'.format(torch.cuda.get_device_name(0)))
    logging.info('Reading config complete.')

    logging.info('Preparing data...')
    input_ids, token_type_ids = tokenize_inputs(claim, evidence, config)
    attention_masks = [float(i > 0) for i in input_ids]
    input_tensor = create_padded_tensor(input_ids, config)
    mask_tensor = create_padded_tensor(attention_masks, config)
    type_tensor = create_padded_tensor(token_type_ids, config)
    label_tensor = torch.tensor([-1], dtype=torch.long)
    logging.info('Preparing data complete.')

    logging.info('Initialising model...')
    model = initialise_model(config)
    logging.info('Initialising model complete.')

    logging.info('Running test...')
    test_preds = test_model(model, input_tensor, type_tensor, mask_tensor)
    logging.info('Test complete.')

    labels_vals = {
        0: 'Refuted',
        1: 'Supported',
        2: 'NotEnoughInfo'
    }

    prediction = labels_vals[test_preds.item()]

    return prediction


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get claim from parameters or request body
    claim = req.params.get('claim')
    if not claim:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            claim = req_body.get('claim')

    # Get evidence from parameters or request body
    evidence = req.params.get('evidence')
    if not evidence:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            evidence = req_body.get('evidence')

    if claim and evidence:
        prediction = predict(claim, evidence)
        return func.HttpResponse(prediction)
    else:
        return func.HttpResponse(
             "Please pass a claim and evidence on the query string or in the request body",
             status_code=400
        )
