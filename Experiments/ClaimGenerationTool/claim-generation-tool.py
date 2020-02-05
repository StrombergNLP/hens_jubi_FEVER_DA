from datetime import datetime
import pandas as pd

source_df = pd.read_csv('data/dawiki-latest-abstract.csv')
source_df = source_df[['title', 'abstract']]
print('{} rows loaded'.format(len(source_df['title'])))

source_df['title'] = source_df['title'].str.lower()

titles = []
abstracts = []
claims = []
dictionaries = []

while True:
    print('Source sentence:')
    source = source_df.sample(1)
    print(source['abstract'].values)
    print('')

    user_input = input("Enter claim or write 'skip' or 'quit'\n")

    if user_input == 'skip':
        continue
    elif user_input == 'quit':
        break

    claim = user_input

    user_input = input("Enter other sources (comma separated) or press enter:\n")
    other_sources = user_input.split(',')
    other_sources = [x.strip() for x in other_sources]

    dictionary = {}
    for os in other_sources:
        if os != '':
            other_row = source_df.query('title == \'{}\''.format(os.lower()))
            if len(other_row['title']):
                other_abstract = ['abstract'].values[0]
                dictionary[os] = other_abstract

    dictionaries.append(dictionary)

    titles.append(source['title'].values[0])
    abstracts.append(source['abstract'].values[0])
    claims.append(claim)

claims_df = pd.DataFrame()
claims_df['claim'] = claims
claims_df['source_title'] = titles
claims_df['dictionary'] = dictionary
claims_df['source_abstract'] = abstracts

claims_df.to_json('out/{}.json'.format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")), orient='records', lines=True)

print('Saved {} claims to file.'.format(len(claims_df['claim'])))



