from datetime import datetime
import pandas as pd
import random

source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = source_df[['title', 'abstract']]
print('{} rows loaded.\n'.format(len(source_df['title'])))

claims_df = pd.DataFrame()

while True:
    source_row = source_df.sample(1) # Sample one row
    source_title = source_row['title'].values[0]
    source_abstract = source_row['abstract'].values[0]
    
    # Sample one sentence from abstract
    abstract_split = source_abstract.split('.') # Too imprecise
    if len(abstract_split):
        source_sentence = random.choice(abstract_split).strip() + '.'
    else:
        continue # Empty abstract, skip

    if len(source_sentence) <= 10:
        continue # Empty/short sentence, skip

    print('Source entity:')
    print(source_title)
    print('Source sentence:')
    print(source_sentence)
    print('')

    user_input = input("Enter claim, hit enter to skip or write 'quit'\n")

    if user_input == 'quit':
        break
    elif user_input != '': # Add some extra validation here
        claims_df = claims_df.append({'claim': user_input, 'evidence': source_sentence, 'entity': source_title}, ignore_index=True)

if not claims_df.empty:
    claims_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} claims to file.'.format(len(claims_df['claim'])))
else:
    print('No claims were saved.')



