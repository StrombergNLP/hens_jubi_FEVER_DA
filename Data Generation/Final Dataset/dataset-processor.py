import pandas as pd

def drop_duplicate_claims(df):
    """ Drops rows with duplicate values in claim column. Modifies DF in place! """
    len_with_dupes = len(df['claim'])
    df.drop_duplicates(subset='claim', inplace=True)
    len_no_dupes = len(df['claim'])
    print('Dropped {} duplicate rows. {} remaining.'.format(len_with_dupes - len_no_dupes, len_no_dupes))
    return df

print('Loading annotations...')
anno_df = pd.read_json('../CommonData/annotations/annotations_00.jsonl', lines=True)
anno_len = len(anno_df.claim)
print('Loaded {} annotations.'.format(anno_len))

anno_df = drop_duplicate_claims(anno_df)
anno_df = anno_df.reset_index(drop=True)
anno_df = anno_df.sample(frac=1)

print('Supported count: {}'.format(len(anno_df.query("label == 'Supported'"))))
print('Refuted count: {}'.format(len(anno_df.query("label == 'Refuted'"))))
print('NotEnoughInfo count: {}'.format(len(anno_df.query("label == 'NotEnoughInfo'"))))

anno_len = len(anno_df.claim)
train_index = int(anno_len * 0.7)
dev_index = train_index + int(anno_len * 0.2)

train_df = anno_df.iloc[:train_index]
train_df.to_json('annotations_train.jsonl', orient='records', lines=True)
dev_df = anno_df.iloc[train_index:dev_index]
dev_df.to_json('annotations_dev.jsonl', orient='records', lines=True)
test_df = anno_df.iloc[dev_index:]
test_df.to_json('annotations_test.jsonl', orient='records', lines=True)