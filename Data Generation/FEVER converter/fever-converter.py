import pandas as pd
import glob

path = r'../CommonData' 
all_DA = glob.glob(path + "/FEVER DA/*.jsonl")
all_EN = glob.glob(path + "/FEVER EN/*.jsonl")

def read_files(all_files):
    li = []
    for filename in all_files:
        df = pd.read_json(filename, lines=True)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    return df

def format_dataframe(df, post_fix_a, post_fix_b):
    df.label = df.label.replace(['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'],['Supported', 'Refuted', 'NotEnoughInfo'])
    df[['label','claim','evidence']] = df[['label','claim' + post_fix_a,'evidence' + post_fix_b]]
    df = df[['label','claim','evidence']]
    return df

def fill_NEI(df):
    filtered_df = df.query('label != "NotEnoughInfo"')
    
    for index, row in df.iterrows():
        if row.label == 'NotEnoughInfo':
            row.evidence = filtered_df.sample(1).evidence.values[0]

    return df

def drop_empty_evidence(df):
    for index, row in df.iterrows():
        if row.evidence == ['']:
            if row.label == 'Supported' or 'Refuted':
                df.drop(index, inplace=True)
        elif row.evidence == ['N/A']:
            df.drop(index, inplace=True)
    return df

da_fever = read_files(all_DA)
en_fever = read_files(all_EN)

da_fever = format_dataframe(da_fever, '_da', '_da')
en_fever = format_dataframe(en_fever, '', '_en')

da_fever = drop_empty_evidence(da_fever)
en_fever = drop_empty_evidence(en_fever)

da_fever = fill_NEI(da_fever)
en_fever = fill_NEI(en_fever)

da_fever.to_json('../CommonData/fever_da.jsonl', orient='records', lines=True)
en_fever.to_json('../CommonData/fever_en.jsonl', orient='records', lines=True)
print('Saved {} danish claims to file.'.format(len(da_fever['claim'])))
print('Saved {} english claims to file.'.format(len(en_fever['claim'])))