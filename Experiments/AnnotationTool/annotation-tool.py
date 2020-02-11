from datetime import datetime
import pandas as pd
import random

# source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = pd.read_json('data/test_data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence']]
print('{} rows loaded.'.format(len(source_df['claim'])))

annotation_df = pd.DataFrame()

while True:
    source_row = source_df.sample(1) # Sample one row
    source_claim = source_row['claim'].values[0]
    source_entity = source_row['entity'].values[0]
    source_evidence = source_row['evidence'].values[0]


    print('\nSource claim:')
    print(source_claim)
    print('Source evidence:')
    print(source_evidence)
    print('')

    user_input = input("[1] Supported \n[2] Refuted \n[3] Not Enough Info \nEnter to skip or 'quit'\n")

    if user_input == 'quit': 
        break

    annotation = ''

    if user_input == '1':
        annotation = 'Supported'

    elif user_input == '2': 
        annotation = 'Refuted'

    elif user_input == '3':
        annotation = 'NotEnoughInfo'
        source_evidence = ''

    if user_input == '': 
        continue
    elif user_input != '':
        annotation_df = annotation_df.append({'claim': source_claim, 'entity': source_entity, 'evidence': source_evidence, 'label': annotation}, ignore_index=True)


if not annotation_df.empty:
    annotation_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} annotations to file.'.format(len(annotation_df['claim'])))

else:
      print('No mutations were saved.')



