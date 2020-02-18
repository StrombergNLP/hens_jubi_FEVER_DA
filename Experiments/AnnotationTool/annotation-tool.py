from datetime import datetime
import pandas as pd
import random

source_df = pd.read_json('data/data-dictionary-small.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence', 'dictionary']]
print('{} rows loaded in claims.'.format(len(source_df['claim'])))

annotation_df = pd.DataFrame()
source_df = source_df.sample(frac=1) # Shuffling the rows 
counter = 1

for index, source_row in source_df.iterrows():
    print('')

    source_claim = source_row['claim']
    source_entity = source_row['entity']
    source_evidence = source_row['evidence']
    source_dict = source_row['dictionary']

    print('------- CLAIM {}/{} -------\n'.format(counter, len(source_df['claim'])))
    counter += 1

    print('Source entity:')
    print(source_entity)
    print('Source claim:')
    print(source_claim)
    print('Source evidence:')
    print(source_evidence)
    print('')

    print('------- DICTIONARY -------\n')

    for index, pair in enumerate(source_dict):
        abstract = pair['abstract']
        print('[{}] {}\n'.format(index, abstract))

    print('-------- ANNOTATE --------\n')

    invalid_input = True

    while invalid_input:      # Keep asking for input until valid
        user_input = input("[S] Supported \n[R] Refuted \n[N] Not Enough Info \nProvide useful sentences by their index \nEnter to skip or 'quit'\n")

        if user_input == 'quit': 
            break

        annotation = ''
        used_sentences = [source_evidence]
        used_entities = [source_entity]
        malformed_input = False

        if user_input == ' ':
            continue 

        input_tokens = user_input.split(' ')
        annotation_token = input_tokens[0]

        if annotation_token.lower() == 's':
            annotation = 'Supported'

        elif annotation_token.lower() == 'r': 
            annotation = 'Refuted'

        elif annotation_token.lower() == 'n':
            annotation = 'NotEnoughInfo'
            source_evidence = ''

        else: 
            malformed_input = True

        for token in input_tokens[1:]:
            token = int(token)
            dict_object = source_dict[token]

            used_entities.append(dict_object['entity'])
            used_sentences.append(dict_object['abstract'])

        annotation_df = annotation_df.append({'claim': source_claim, 'entity': used_entities, 'evidence': used_sentences, 'label': annotation}, ignore_index=True)


else:
    print('No more claim + evidence pairs to annotate. Good job, Old Sport!')

if not annotation_df.empty:
    annotation_df.to_json('out/{}.jsonl'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)
    print('Saved {} annotations to file.'.format(len(annotation_df['claim'])))

else:
      print('No annutations were saved.')



