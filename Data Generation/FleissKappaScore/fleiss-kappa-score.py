from statsmodels.stats.inter_rater import fleiss_kappa
import pandas as pd
import krippendorff

annotator1_df = pd.read_json('data/da-agreement-julie.jsonl', lines=True)
annotator2_df = pd.read_json('data/da-agreement-henri.jsonl', lines=True)

combined_df = annotator1_df.join(annotator2_df, rsuffix='2', lsuffix='1')
keep = ['claim1', 'label1', 'label2']
combined_df = combined_df[keep]

results_df = pd.DataFrame()

for index, row in combined_df.iterrows():
    supported_val = 0
    refuted_val = 0
    nei_val = 0

    for label in ['label1', 'label2']:
        value = row[label]

        if value == 'Supported':
            supported_val +=1
        elif value == 'Refuted': 
            refuted_val += 1
        else:
            nei_val +=1

    results_df = results_df.append({'Supported': supported_val, 'Refuted': refuted_val, 'NotEnoughInfo': nei_val}, ignore_index=True)

results_alpha_df = pd.DataFrame()

for index, row in combined_df.iterrows():
    claim = row['claim1']
    annotator_1 = row['label1']
    annotator_2 = row['label2']

    if annotator_1 == 'Supported':
        annotator_1 = 1
    elif annotator_1 == 'Refuted':
        annotator_1 = 2
    else: 
        annotator_1 = 3

    if annotator_2 == 'Supported':
        annotator_2 = 1
    elif annotator_2 == 'Refuted':
        annotator_2 = 2
    else: 
        annotator_2 = 3

    alpha_val = [annotator_1, annotator_2]
    results_alpha_df[claim] = alpha_val

kappa = fleiss_kappa(results_df, method='fleiss')
alpha = krippendorff.alpha(results_alpha_df)    # -> https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

print('K-Score: {:1.3f}'.format(kappa))
print('Krippendorf-Score: {:1.3f}'.format(alpha))