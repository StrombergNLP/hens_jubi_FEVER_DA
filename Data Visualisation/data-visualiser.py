import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#---------
# Methods
#---------

def parse_data(data_path):
    data_df = pd.read_json(data_path, lines=True)
    config = data_df.config.values[0]
    c_matrix = data_df.confusion_matrix.values[0]
    loss = data_df.loss.values[0]
    micro_f1, macro_f1 = data_df.micro_f1.values[0], data_df.macro_f1.values[0]

    return config, c_matrix, loss, micro_f1, macro_f1

def plot_confusion_matrix(c_matrix, num_labels):
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

#---------
# Main
#---------

config, c_matrix, loss, micro_f1, macro_f1 = parse_data('data/11-03-2020-14-31-25.json')
plot_confusion_matrix(c_matrix, config['num_labels'])
plot_loss(loss)