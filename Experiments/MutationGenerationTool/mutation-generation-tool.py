from datetime import datetime
import pandas as pd
import random

# source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = pd.read_json('data/test_data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence']]
print('{} rows loaded.\n'.format(len(source_df['claim'])))

mutations_df = pd.DataFrame()


while True:
    source_row = source_df.sample(1) # Sample one row
    source_claim = source_row['claim'].values[0]
    source_entity = source_row['entity'].values[0]
    source_evidence = source_row['evidence'].values[0]

    # Save original claim 
    mutations_df = mutations_df.append({'claim': source_claim, 'entity': source_entity, 'evidence': source_evidence}, ignore_index=True)
     
    print('Source claim:')
    print(source_claim + '\n')

    while True:
        user_input = input("Write mutation or hit enter to skip) \n")

        if user_input == '':
            print('')
            break
        else:
            user_input = user_input.strip()    
            mutations_df = mutations_df.append({'claim': user_input, 'entity': source_entity, 'evidence': source_evidence}, ignore_index=True)
        print('')

    user_input = input("Hit enter for new claim or 'quit' \n")
    
    if user_input == 'quit': 
        break
    else: 
        continue


if not mutations_df.empty:
    mutations_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} mutations to file.'.format(len(mutations_df['claim'])))
else:
     print('No mutations were saved.')



