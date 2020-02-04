# Code from: https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translate?pivots=programming-language-python

# -*- coding: utf-8 -*-
import os, requests, uuid, json
import pandas as pd

# Extract key and endpoint from environment variables
key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

# Set up parameters
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
data = []
with open('data/train.jsonl') as train_json_file:
    for line in train_json_file:
        line_decoded = json.loads(line)
        data.append(line_decoded)
    
# Convert to dataframe
df = pd.DataFrame(data)

# Sample
df = df.sample(3)

# Loop over claims and translate each
claims_translated = []
for index, row in df.iterrows():
    claim = row['claim']

    # Text to translate
    body = [{
        'text': claim
    }]

    # Make a POST request
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    # Save translation to list
    claim_da = response[0]['translations'][0]['text']
    claims_translated.append(claim_da)

# Save translations to dataframe
df.insert(4, 'claim_da', claims_translated)

print(df)