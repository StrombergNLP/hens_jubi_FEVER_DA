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

def plot_single_box(f1_score, f1_type):
    sn.set(style="whitegrid", font_scale=0.75)
    f1_score[f1_type] = f1_score[0]
    ax = sn.boxplot(x=f1_score[f1_type], orient='v', width=0.35, palette='Blues_d')
    plt.show()

def plot_multi_box(micro, macro):
    combined_df = pd.DataFrame(columns=['micro', 'macro'])
    combined_df['micro'] = micro[0]
    combined_df['macro'] = macro[0]
    sn.set(style="whitegrid", font_scale=0.75)
    ax = sn.boxplot(data=combined_df, order=['micro', 'macro'], width=0.35, palette='Blues_d')
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


#---------
# Main
#---------

# Order: [config, c_matrix, loss, micro_f1, macro_f1]
results_1 = parse_data('out/17-03-2020-14-00-35.json')
results_2 = parse_data('out/17-03-2020-13-55-47.json')
results_3 = parse_data('out/17-03-2020-13-52-00.json')
results_4 = parse_data('out/17-03-2020-13-44-23.json')
results_5 = parse_data('out/17-03-2020-13-33-09.json')

# Testing defs
# plot_confusion_matrix(results_1[1], results_1[0])
# plot_loss(results_1[2])
# avg_loss([results_1[2], results_3[2]])
# avg_matrix([results_1[1], results_3[1], results_2[1], results_4[1], results_5[1]], results_1[0]) 
micro_df = pd.DataFrame([results_1[3], results_3[3], results_2[3], results_4[3], results_5[3]])
macro_df = pd.DataFrame([results_1[4], results_3[4], results_2[4], results_4[4], results_5[4]])
# plot_single_box(micro_df, 'Micro F1')
# plot_single_box(macro_df, 'Macro F1')
# plot_multi_box(micro_df, macro_df)
loss_df = pd.DataFrame(columns=['lr1', 'lr2'])
loss_df['lr1'] = results_1[2]
loss_df['lr2'] = results_3[2]
plot_multi_loss(loss_df)