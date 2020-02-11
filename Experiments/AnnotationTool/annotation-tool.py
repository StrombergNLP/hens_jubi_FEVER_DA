from datetime import datetime
import pandas as pd
import random

# source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = pd.read_json('data/data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence']]
print('{} rows loaded.'.format(len(source_df['claim'])))

annotation_df = pd.DataFrame()

source_df = source_df.sample(frac=1) # Shuffling the rows 

for index, source_row in source_df.iterrows():
    source_claim = source_row['claim']
    source_entity = source_row['entity']
    source_evidence = source_row['evidence']

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

print('No more claim + evidence pairs to annotate. Good job, Old Sport!')

if not annotation_df.empty:
    annotation_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} annotations to file.'.format(len(annotation_df['claim'])))

else:
      print('No mutations were saved.')



