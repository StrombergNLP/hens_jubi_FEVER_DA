from statsmodels.stats.inter_rater import fleiss_kappa
import pandas as pd
import krippendorff

annotator1_df = pd.read_json('data/test-annotations1.jsonl', lines=True)
annotator2_df = pd.read_json('data/test-annotations2.jsonl', lines=True)

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

kappa = fleiss_kappa(results_df, method='fleiss')
# alpha = krippendorff.alpha(results_df)    # -> https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

print('K-Score: {:1.3f}'.format(kappa))