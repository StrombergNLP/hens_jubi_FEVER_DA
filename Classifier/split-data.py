import pandas as pd 
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/replaced_content_words.jsonl'
data_df = pd.read_json(DATA_PATH, lines=True)
train_df,validation_df = train_test_split(data_df, test_size=0.1)

train_df.to_json('data/train_data_replaced_content_words.jsonl', orient='records', lines=True)
validation_df.to_json('data/validation_data_replaced_content_words.jsonl', orient='records', lines=True)