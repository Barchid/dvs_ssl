from torchvision import transforms
from project.utils.cka import CKA
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import torch
import os
from itertools import combinations


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="plasma", cbar_kws={'label': 'Scale'})
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="Method", xlabel="Method")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    

def cka_cm(
    stats, key="embeddings"
):
    data = [[0]] * len(stats)
    
    name2idx = {}
    labels = []
    for i, stat in enumerate(stats):
        name = stat[0]
        labels.append(name)
        name2idx[name] = i
            
    cka_cool = CKA()
    # res = list(combinations(test_list, 2))
    
    features = []
    # fill data
    for stat in stats:
        features.append(stat)
    
    pairs = list(combinations(features, 2))
    
    for pair in pairs:
        name_X = pair[0][0]
        name_Y = pair[1][0]
        X = pair[0][1][key]
        Y = pair[1][1][key]
        
        cka_val = cka_cool.linear_CKA(X, Y)
        i = name2idx[name_X]
        j = name2idx[name_Y]
        
        data[i][j] = cka_val.item()
    
    print(labels)
    print(data)
    return data


def main():
    path = "/sandbox0/sami/stats"
    
    stats = []
    for i, stat_dir in enumerate(os.listdir(path)):
        if "dvsgesture" not in stat_dir:
            continue
        
        stat_dir_path = os.path.join(path, stat_dir)
        
        enc1: str = stat_dir.split("_")[1]
        enc2 = stat_dir.split("_")[2]
        
        enc_name = f"{enc1.upper()}"
        if enc2 == "supervised":
            enc_name += "_sup"
        elif enc1 != enc2 and enc1 == "snn":
            if enc2 == "3dcnn":
                enc_name += "_3d"
            else:
                enc_name += "_cnn"
        elif enc1 != enc2 and enc1 != "snn":
            enc_name += "_snn"
            
        stat_file = os.path.join(stat_dir_path, "cka_stats.pt")
        
        stats.append(
            (enc_name, torch.load(stat_file))
        )
        
    cka_cm(stats)
        
    

if __name__ == "__main__":
    main()