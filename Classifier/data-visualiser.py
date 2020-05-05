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

def plot_confusion_matrix(c_matrix, num_labels):
    df_cm = pd.DataFrame(c_matrix, range(num_labels), range(num_labels))
    df_cm.columns = ['Refuted', 'Supported', 'NotEnoughInfo']
    df_cm['Labels'] = ['Refuted', 'Supported', 'NotEnoughInfo']
    df_cm = df_cm.set_index('Labels')
    plt.figure(figsize=(10,7))
    sn.set(style='white', font_scale=0.75)
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

def plot_bars(df, xaxis, yaxis, column_name=None, xlabel=None, ylabel=None, hue=None, ):
    sn.set(style='white', font_scale=0.75)
    ax = sn.barplot(x=xaxis, y=yaxis, hue=hue, data=df, palette='muted')
    # ax = sn.catplot(x=xaxis, y=yaxis, hue=hue, col=column_name, data=df, kind='bar', aspect=.7, palette='muted')
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    plt.show()

def scatter_plot(df, xaxis, yaxis):
    sn.set(style="whitegrid")
    ax = sn.scatterplot(x=xaxis, y=yaxis, data=df, palette='Blues_d')
    plt.show()

def plot_precision_recall(c_matrix, model_type):
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
    
    precision_recall_df = pd.DataFrame(columns=['model type', 'label', 'attribute', 'value'])
    precision_recall_df['label'] = ['Refuted', 'Refuted', 'Supported', 'Supported', 'NEI', 'NEI']
    precision_recall_df['attribute'] = ['precision', 'recall', 'precision', 'recall', 'precision', 'recall']
    precision_recall_df['value'] = [refuted_precision, refuted_recall, supported_precision, supported_recall, nei_precision, nei_recall]
    precision_recall_df['Balancing strategy'] = [model_type, model_type, model_type, model_type, model_type, model_type]

    return precision_recall_df

def calculate_precision_recall(true_positives, false_positives, false_negatives):
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0

    try: 
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0

    return precision, recall 

def melt_dataframe(df, varsname):
    df = df.melt(id_vars=varsname, var_name='attribute')
    return df

def plot(df):
    sn.set(style='white', font_scale=0.75)
    ax = sn.lineplot(x="l, k", y="value", hue="attribute", data=df, 
            style="attribute", markers=True, dashes=False, palette="muted")
    plt.show()
    #sns.lineplot(x="timepoint", y="signal", data=fmri)

#---------
# model epoch
#---------
data_df = pd.DataFrame(columns=['label', 'micro $f_1$', 'macro $f_1$'])
data_df['label'] = ['k= 3, l= 10', 'k= 1, l= 4', 'Original']
data_df['micro $f_1$'] = [0.564, 0.596, 0.672]
data_df['macro $f_1$'] = [0.469, 0.546, 0.629]
data_df = melt_dataframe(data_df, 'label')
plot_bars(data_df, 'value', 'label', hue='attribute', ylabel=' ')

#---------
# k,l lines
#---------
# data_df = pd.DataFrame(columns=['l, k', 'precision', 'recall', 'f1'])
# data_df['l, k'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# data_df['precision'] = [0.693, 0.54, 0.460, 0.380, 0.335, 0.286, 0.256, 0.234, 0.214, 0.195]
# data_df['recall'] = [0.116, 0.172, 0.211, 0.231, 0.253, 0.258, 0.268, 0.279, 0.284, 0.288]
# data_df['f1'] = [0.198, 0.261, 0.290, 0.288, 0.288, 0.272, 0.262, 0.254, 0.244, 0.233]
# data_df = melt_dataframe(data_df, 'l, k')
# print(data_df)
# plot(data_df)

#---------
# Exp
#---------

# labeled_paths = {
#     'base': 'results/exp/30-04-2020-06-27-28.json',
#     'classweights': 'results/exp/30-04-2020-06-38-12.json',
#     'nothing': 'results/exp/30-04-2020-06-46-42.json',
#     'oversampling': 'results/exp/30-04-2020-07-13-11.json'
# }

# exp_df = read_files(labeled_paths)
# none_df = plot_precision_recall(exp_df.query("label == 'nothing'").confusion_matrix.values[0], 'No balancing')
# class_weights_df = plot_precision_recall(exp_df.query("label == 'classweights'").confusion_matrix.values[0], 'Class weights only')
# base_df = plot_precision_recall(exp_df.query("label == 'base'").confusion_matrix.values[0], 'Oversampling + class weights')
# oversampling_df = plot_precision_recall(exp_df.query("label == 'oversampling'").confusion_matrix.values[0], 'Oversampling only')
# combined = none_df.append(base_df).append(class_weights_df).append(oversampling_df)
# # plot_bars(combined, 'label', 'value', 'Balancing strategy', hue='attribute', xlabel='label')

# exp_df = exp_df[['label', 'micro_f1', 'macro_f1']]
# exp_df = melt_dataframe(exp_df)
# exp_df = exp_df.replace('macro_f1', 'macro $f_1$')
# exp_df = exp_df.replace('micro_f1', 'micro $f_1$')
# exp_df = exp_df.replace('base', 'Oversampling\n+ class weights')
# exp_df = exp_df.replace('nothing', 'No balancing')
# exp_df = exp_df.replace('classweights', 'Class weights only')
# exp_df = exp_df.replace('oversampling', 'Oversampling only')
# plot_bars(exp_df, 'label', 'value', hue='attribute', xlabel='Balancing strategy', ylabel='$f_1$ score') 

#---------
# Development set 
#---------

# labeled_paths = {
#     'dev_set': 'results/train_28-04-2020-16-20-03.json'
# }
# dev_df = read_files(labeled_paths)
# print(dev_df.confusion_matrix.values[0])
# plot_confusion_matrix(dev_df.confusion_matrix.values[0], 3)

#---------
# Train loss over learning rates
#---------

# labeled_paths = {
#     # '1e-6': 'results/lrdata/1e-6.json', 
#     '1e-5': 'results/lrdata/1e-5.json',
#     '2e-5': 'results/lrdata/2e-5.json',
#     '3e-5': 'results/lrdata/3e-5.json',
#     # '1e-4': 'results/lrdata/1e-4.json'
# }

# lr_df = read_files(labeled_paths)
# lr_df = lr_df[['label', 'micro_f1', 'macro_f1']]
# lr_df['micro $f_1$'] = lr_df['micro_f1']
# lr_df['macro $f_1$'] = lr_df['macro_f1']
# lr_df = lr_df[['label', 'micro $f_1$', 'macro $f_1$']]
# print(lr_df)
# lr_df = melt_dataframe(lr_df, 'label')
# plot_bars(lr_df, 'label', 'value', xlabel='learning rate', ylabel='$f_1$ score', hue='attribute')

# lr_df['learning rate'] = lr_df['label']
# lr_df = lr_df[['learning rate', 'loss']]
# print(lr_df)
# # plot_single_box(lr_df)
# plot_multi_box(lr_df, ['1e-6', '1e-5', '2e-5', '3e-5', '1e-4'])

#---------
# Ablations
#---------

# Random labels

# labeled_paths = {
#     0.0: 'results/24-03-2020-13-24-58.json',
#     0.1: 'results/25-03-2020-13-19-45.json',
#     0.5: 'results/25-03-2020-13-36-03.json',
#     1.0: 'results/25-03-2020-13-52-21.json'
# }
# random_labels_df = read_files(labeled_paths)
# # plot_precision_recall(random_labels_df.query("label == 1.0").confusion_matrix.values[0])

# random_labels_df = melt_dataframe(random_labels_df[['label', 'micro_f1', 'macro_f1']])
# random_labels_df = random_labels_df.replace('micro_f1', 'micro $f_1$')
# random_labels_df = random_labels_df.replace('macro_f1', 'macro $f_1$')
# random_labels_df = random_labels_df.replace('1.0', '100')
# plot_bars(random_labels_df, 'label', 'value', hue='attribute', xlabel='random labels %', ylabel='$f_1$ score') 

# # MAX LEN (incomplete data)
# labeled_paths = {
#     5: 'results/25-03-2020-12-29-53.json',
#     25: 'results/25-03-2020-12-22-29.json',
#     75: 'results/25-03-2020-15-05-39.json',
#     125: 'results/25-03-2020-14-48-23.json',
#     250: 'results/24-03-2020-13-24-58.json',
# }

# maxlen_df = read_files(labeled_paths)
# maxlen_df = melt_dataframe(maxlen_df[['label', 'micro_f1', 'macro_f1']])
# maxlen_df = maxlen_df.replace('micro_f1', 'micro $f_1$')
# maxlen_df = maxlen_df.replace('macro_f1', 'macro $f_1$')
# # plot_bars(maxlen_df, 'label', 'value', hue='attribute', xlabel='max len', ylabel='$f_1$ score')    

# # Shuffled word order
# labeled_paths = {
#     'base': 'results/24-03-2020-13-24-58.json',
#     'shuffled': 'results/25-03-2020-12-53-56.json',
#     'no evidence': 'results/26-03-2020-08-34-50.json'
# }

# shuffled_words_df = read_files(labeled_paths)
# shuffled_words_df = melt_dataframe(shuffled_words_df[['label', 'micro_f1', 'macro_f1']])
# shuffled_words_df = shuffled_words_df.replace('micro_f1', 'micro $f_1$')
# shuffled_words_df = shuffled_words_df.replace('macro_f1', 'macro $f_1$')
# # plot_bars(shuffled_words_df, 'label', 'value', hue='attribute', ylabel='$f_1$ score', xlabel='data ablation')

# # Replacing verbs and nouns
# labeled_paths = {
#     'base': 'results/24-03-2020-13-24-58.json',
#     'replaced words': 'results/26-03-2020-07-15-30.json'
# }
# replaced_words_df = read_files(labeled_paths)
# replaced_words_df = melt_dataframe(replaced_words_df[['label', 'micro_f1', 'macro_f1']])
# replaced_words_df = replaced_words_df.replace('micro_f1', 'micro $f_1$')
# replaced_words_df = replaced_words_df.replace('macro_f1', 'macro $f_1$')
# # plot_bars(replaced_words_df, 'label', 'value', hue='attribute', ylabel='$f_1$ score', xlabel='data ablation')