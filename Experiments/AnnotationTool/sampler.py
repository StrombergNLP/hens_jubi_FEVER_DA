from datetime import datetime
import pandas as pd

source_df = pd.read_json('data/data.jsonl', lines=True)
source_df = source_df[['claim', 'entity', 'evidence']]
source_df = source_df.sample(100)

source_df.to_json('data/da-agreement-sample.jsonl', orient='records', lines=True)