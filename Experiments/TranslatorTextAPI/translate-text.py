# Code from: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translate?pivots=programming-language-python

# -*- coding: utf-8 -*-
import os, requests, uuid, json
from datetime import datetime
import pandas as pd

def translate(text):
    """Translate given text to Danish. Return translated text."""

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
with open('data/train_04_feb_test.jsonl') as train_json_file:
    for line in train_json_file:
        line_decoded = json.loads(line)
        data.append(line_decoded)

# Read evidence
print('Reading evidence...')
evidence = []
with open('data/evidence_04_feb_test.jsonl') as evidence_json_file:
    for line in evidence_json_file:
        line_decoded = json.loads(line)
        evidence.append(line_decoded)
    
# Convert to dataframe
train_df = pd.DataFrame(data)
evidence_df = pd.DataFrame(evidence)

# Sample
#train_df = train_df.sample(3)

# Loop over claims and translate each
print('Processing claims...')
claims_translated = []
for index, row in train_df.iterrows():
    # Claim
    claim = row['claim']

    claim_da = translate(claim)
    claims_translated.append(claim_da)

    # Evidence

    # # Lookup evidence
    # claim_evidence_list = row['evidence'][0]
    # for claim_evidence in claim_evidence_list:
    #     evidence_id = claim_evidence[2]
    #     evidence_row = evidence_df.query('id == \'{}\''.format(evidence_id))
    #     evidence_text = evidence_row['text'].values[0]

    #     evidence_text_da = translate(evidence_text)
    # TODO: Decide what to do with the translated evidence (for now)

# Save translations to dataframe
train_df.insert(4, 'claim_da', claims_translated)
# print(train_df[['claim', 'claim_da']])

# Save dataframe as csv
train_df.to_csv('out/{}.csv'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))

print("SUCCESS!")