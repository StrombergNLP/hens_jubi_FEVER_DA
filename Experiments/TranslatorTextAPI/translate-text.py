# Code from: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translate?pivots=programming-language-python

# -*- coding: utf-8 -*-
import os, requests, uuid, json
from datetime import datetime
import pandas as pd

def translate(text):
    """Translate given text to Danish. Return translated text."""
    print('Translating {}...'.format(text[:15]))

     # Set request body for request
    body = [{
        'text': text
    }]

    # Make the POST request
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    return response[0]['translations'][0]['text'] # navigating through json object response

# Extract key and endpoint from environment variables
print('Reading environment variables...')
key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

# Set up parameters
print('Setting translation parameters...')
path = '/translate?api-version=3.0'
params = '&from=en&to=da'
constructed_url = endpoint + path + params

# Pass subscription key for authentication
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# Read dataset
print('Reading training data...')
data = []
with open('data/train.jsonl') as train_json_file:
    for line in train_json_file:
        line_decoded = json.loads(line)
        data.append(line_decoded)

# Read rosetta stone
print('Reading Rosetta Stone...')
rosetta_df = pd.read_json('rosetta-stone.json', lines=True)

# Read index
print('Reading index...')
index_df = pd.read_json('wiki-index.jsonl', lines=True)
    
# Convert to dataframe
train_df = pd.DataFrame(data)

# Sample
train_df = train_df.sample(3)

# Loop over claims and translate each
print('Processing claims...')
claims_translated = []
evidences_translated = []
evidences_en = []

i = 1
for index, row in train_df.iterrows():
    print('Processing claim {}/{}...'.format(i, len(train_df['id']+1)), end='\r')
    i += 1
    # Claim
    claim_en = row['claim']

    claim_da = translate(claim_en)
    claims_translated.append(claim_da)

    # Evidence

    # Lookup evidence
    claim_evidence_list = row['evidence'][0]
    claim_evidence_translated = []
    claim_evidence_en = []

    for claim_evidence in claim_evidence_list:
        evidence_id = claim_evidence[2]
        if evidence_id is None:
            evidence_da = ''
            evidence_en = ''
        else:
            # Determine whether we already translated it before
            translation_row = rosetta_df.query('id == \'{}\''.format(evidence_id))
            if len(translation_row['id']):
                evidence_da = translation_row['evidence_da'].values[0]
                evidence_en = translation_row['evidence_en'].values[0]
            else:

                # Look up file name in index
                index_row = index_df.query('id == \'{}\''.format(evidence_id))
                file_name = index_row['file_name'].values[0]

                # Look up evidence in file
                wiki_file = pd.read_json('data/wiki-pages/' + file_name, lines=True)
                evidence_row = wiki_file.query('id == \'{}\''.format(evidence_id))
                evidence_en = evidence_row['text'].values[0]

                # Translate evidence
                evidence_da = translate(evidence_en)

                rosetta_df = rosetta_df.append({'id': evidence_id, 'evidence_da': evidence_da, 'evidence_en': evidence_en}, ignore_index=True)

        claim_evidence_translated.append(evidence_da)
        claim_evidence_en.append(evidence_en)
    evidences_translated.append(claim_evidence_translated)
    evidences_en.append(claim_evidence_en)
print('')

# Save translations to dataframe
train_df.insert(4, 'claim_da', claims_translated)
train_df.insert(5, 'evidence_da', evidences_translated)
train_df.insert(6, 'evidence_en', evidences_en)

print('Saving claims and evidence...')
# Save dataframe as json
train_df_en = train_df[['id', 'label', 'claim', 'evidence_en']]
train_df_en.to_json('out/{}_en.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
train_df_da = train_df[['id', 'label', 'claim_da', 'evidence_da']]
train_df_da.to_json('out/{}_da.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)

print('Saving rosetta stone...')
# Save rosetta stone
rosetta_df.to_json('rosetta-stone.json', orient='records', lines=True)

print("SUCCESS!")