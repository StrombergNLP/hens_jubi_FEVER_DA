import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

#---------
# Methods
#---------

def parse_data(data_path):
    data_df = pd.read_json(data_path, lines=True)
    config = data_df.config.values[0]
    c_matrix = data_df.confusion_matrix.values[0]
    loss = data_df.loss.values[0]
    micro_f1, macro_f1 = data_df.micro_f1.values[0], data_df.macro_f1.values[0]

    return [config, c_matrix, loss, micro_f1, macro_f1]

def read_files(labeled_paths):
    df = pd.DataFrame(columns=['label', 'config', 'confusion_matrix', 'loss', 'micro_f1', 'macro_f1'])
    for label, path in labeled_paths.items():
        content = pd.read_json(path, lines=True)
        content['label'] = label
        df = df.append(content)
    return df

def plot_confusion_matrix(c_matrix, config):
    num_labels = config['num_labels']
    df_cm = pd.DataFrame(c_matrix, range(num_labels), range(num_labels))
    df_cm.columns = ['R', 'S', 'NEI']
    df_cm['Labels'] = ['R', 'S', 'NEI']
    df_cm = df_cm.set_index('Labels')
    plt.figure(figsize=(10,7))
    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={'size':10})    # font size
    plt.show()

def plot_loss(train_loss):
    """ Plot loss after training the model"""
    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss)
    plt.show()

def plot_multi_loss(combined_data):
    ax = sn.lineplot(data=combined_data)
    plt.show()

def plot_single_box(df):
    sn.set(style="whitegrid", font_scale=0.75)
    ax = sn.boxplot(x=df.label.head(1), y=df.value.head(1), orient='v', width=0.35, palette='Blues_d')
    plt.show()

def plot_multi_box(df, order):
    sn.set(style='white', font_scale=0.75)
    ax = sn.boxplot(data=df, x='learning rate', y='loss', order= order, orient='v', palette='muted')
    plt.show()

def avg_loss(combined_data):
    """ 
    Gets the avg of train loss per row. 
    List of loss must be same length. 
    Plots the average loss.
    """
    average_loss = np.average(combined_data, axis=0)
    plot_loss(average_loss)

def avg_matrix(combined_data, config):
    """ 
    Takes multiple matrices, gets the average of predictions.
    Plots the average prediction. 
    """
    average_matrix = np.average(combined_data, axis=0)
    plot_confusion_matrix(average_matrix, config)

def plot_bars(df, xaxis, yaxis, xlabel=None, ylabel=None, hue=None):
    sn.set(style='white', font_scale=0.75)
    ax = sn.barplot(x=xaxis, y=yaxis, hue=hue, data=df, palette='muted')
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    plt.show()

def scatter_plot(df, xaxis, yaxis):
    sn.set(style="whitegrid")
    ax = sn.scatterplot(x=xaxis, y=yaxis, data=df, palette='Blues_d')
    plt.show()

def plot_precision_recall(c_matrix):
    refuted, supported, nei = c_matrix[0], c_matrix[1], c_matrix[2]

    true_positives = refuted[0]
    false_negatives = refuted[1] + refuted[2]
    false_positives = supported[0] + nei[0]
    refuted_precision, refuted_recall = calculate_precision_recall(true_positives, false_positives, false_negatives)

    true_positves = supported[1]
    false_negatives = supported[0] + supported[2]
    false_positives = refuted[1] + nei[1]
    supported_precision, supported_recall = calculate_precision_recall(true_positives, false_positives, false_negatives)

    true_positives = nei[2]
    false_negatives = nei[0] + nei[1]
    false_positives = refuted[2] + supported[2]
    nei_precision, nei_recall = calculate_precision_recall(true_positives, false_positives, false_negatives)
    
    precision_recall_df = pd.DataFrame(columns=['label', 'attribute', 'value'])
    precision_recall_df['label'] = ['Refuted', 'Refuted', 'Supported', 'Supported', 'NEI', 'NEI']
    precision_recall_df['attribute'] = ['precision', 'recall', 'precision', 'recall', 'precision', 'recall']
    precision_recall_df['value'] = [refuted_precision, refuted_recall, supported_precision, supported_recall, nei_precision, nei_recall]

    plot_bars(precision_recall_df, 'label', 'value', hue='attribute', xlabel='label')

def calculate_precision_recall(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall 

def melt_dataframe(df):
    df = df.melt(id_vars='label', var_name='attribute')
    return df

#---------
# Train loss over learning rates
#---------
# labeled_paths = {
#     '1e-6': 'results/lrdata/1e-6.json', 
#     '1e-5': 'results/lrdata/1e-5.json',
#     '2e-5': 'results/lrdata/2e-5.json',
#     '3e-5': 'results/lrdata/3e-5.json',
#     '1e-4': 'results/lrdata/1e-4.json'
# }

# lr_df = read_files(labeled_paths)
# lr_df['learning rate'] = lr_df['label']
# lr_df = lr_df[['learning rate', 'loss']]
# print(lr_df)
# # plot_single_box(lr_df)
# plot_multi_box(lr_df, ['1e-6', '1e-5', '2e-5', '3e-5', '1e-4'])

#---------
# Ablations
#---------

# Random labels

labeled_paths = {
    0.0: 'results/24-03-2020-13-24-58.json',
    0.1: 'results/25-03-2020-13-19-45.json',
    0.5: 'results/25-03-2020-13-36-03.json',
    1.0: 'results/25-03-2020-13-52-21.json'
}
random_labels_df = read_files(labeled_paths)
# plot_precision_recall(random_labels_df.query("label == 0.5").confusion_matrix.values[0])

random_labels_df = melt_dataframe(random_labels_df[['label', 'micro_f1', 'macro_f1']])
random_labels_df = random_labels_df.replace('micro_f1', 'micro $f_1$')
random_labels_df = random_labels_df.replace('macro_f1', 'macro $f_1$')
# plot_bars(random_labels_df, 'label', 'value', hue='attribute', xlabel='random labels %', ylabel='$f_1$ score') 

# MAX LEN (incomplete data)
labeled_paths = {
    5: 'results/25-03-2020-12-29-53.json',
    25: 'results/25-03-2020-12-22-29.json',
    75: 'results/25-03-2020-15-05-39.json',
    125: 'results/25-03-2020-14-48-23.json',
    250: 'results/24-03-2020-13-24-58.json',
}

maxlen_df = read_files(labeled_paths)
maxlen_df = melt_dataframe(maxlen_df[['label', 'micro_f1', 'macro_f1']])
maxlen_df = maxlen_df.replace('micro_f1', 'micro $f_1$')
maxlen_df = maxlen_df.replace('macro_f1', 'macro $f_1$')
# plot_bars(maxlen_df, 'label', 'value', hue='attribute', xlabel='max len', ylabel='$f_1$ score')    

# Shuffled word order
labeled_paths = {
    'base': 'results/24-03-2020-13-24-58.json',
    'shuffled': 'results/25-03-2020-12-53-56.json',
    'no evidence': 'results/26-03-2020-08-34-50.json'
}

shuffled_words_df = read_files(labeled_paths)
shuffled_words_df = melt_dataframe(shuffled_words_df[['label', 'micro_f1', 'macro_f1']])
shuffled_words_df = shuffled_words_df.replace('micro_f1', 'micro $f_1$')
shuffled_words_df = shuffled_words_df.replace('macro_f1', 'macro $f_1$')
# plot_bars(shuffled_words_df, 'label', 'value', hue='attribute', ylabel='$f_1$ score', xlabel='data ablation')

# Replacing verbs and nouns
labeled_paths = {
    'base': 'results/24-03-2020-13-24-58.json',
    'replaced words': 'results/26-03-2020-07-15-30.json'
}
replaced_words_df = read_files(labeled_paths)
replaced_words_df = melt_dataframe(replaced_words_df[['label', 'micro_f1', 'macro_f1']])
replaced_words_df = replaced_words_df.replace('micro_f1', 'micro $f_1$')
replaced_words_df = replaced_words_df.replace('macro_f1', 'macro $f_1$')
plot_bars(replaced_words_df, 'label', 'value', hue='attribute', ylabel='$f_1$ score', xlabel='data ablation')
