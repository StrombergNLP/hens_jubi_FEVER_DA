from datetime import datetime
import pandas as pd
import random, nltk

source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = source_df[['title', 'abstract']]
print('{} rows loaded.'.format(len(source_df['title'])))

claims_df = pd.DataFrame()
counter = 1

while True:
    source_row = source_df.sample(1) # Sample one row
    source_title = source_row['title'].values[0]
    source_abstract = source_row['abstract'].values[0]
    
    # Split abstract into sentences (better than just splitting by '.')
    source_abstract_tokenized = nltk.sent_tokenize(source_abstract, language='danish')
    if len(source_abstract_tokenized):
        # Sample one sentence from abstract
        source_sentence = random.choice(source_abstract_tokenized)
    else:
        continue # Empty abstract, skip

    if len(source_sentence) <= 20 or not source_sentence.endswith('.'):
        continue # Empty/short sentence, skip

    print('\nYou are working on claim number: {} \n'.format(counter))

    print('Source entity:')
    print(source_title)
    print('Source sentence:')
    print(source_sentence)
    print('')

    user_input = input("Enter claim, hit enter to skip or write 'quit'\n")

    if user_input == 'quit':
        break
    elif user_input != '': # Add some extra validation here
        counter +=1
        claims_df = claims_df.append({'claim': user_input, 'evidence': source_sentence, 'entity': source_title}, ignore_index=True)

if not claims_df.empty:
    claims_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} claims to file.'.format(len(claims_df['claim'])))
else:
    print('No claims were saved.')