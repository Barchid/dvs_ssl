import torch
import json
import numpy as np
import os
from argparse import ArgumentParser
from project.utils.uniform_loss import uniformity, tolerance
from project.utils.cka import CKA, CudaCKA
import seaborn
import matplotlib.pyplot as plt

 
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
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
 
# define data
data = [[13, 1, 1, 0, 2, 0],
        [3, 9, 6, 0, 1, 0],
        [0, 0, 16, 2, 0, 0],
        [0, 0, 0, 13, 0, 0],
        [0, 0, 0, 0, 15, 0],
        [0, 0, 1, 0, 0, 15]]
 
# define labels
labels = ["A", "B", "C", "D", "E", "F"]
 
# create confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")


def main(
    embeddings,
):
    np.set_printoptions(precision=2)
    
    cka_cool = CKA('cpu')
    
    labels = []
    for (method, _) in embeddings:
        labels.append(method)
        
    cm = [[0]] * len(embeddings)
    
    min_val = 0.
    max_val = 1.
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            method1, filename1 = embeddings[i]
            method2, filename2 = embeddings[j]
            embedding1 = torch.load(filename1)
            embedding2 = torch.load(filename2)
            
            print(f"Computing CKA value between {method1} and {method2}")
            cka_val = cka_cool.linear_CKA(embedding1, embedding2)
            cm[i][j] = cka_val
            
            if cka_val > max_val:
                max_val = cka_val
            
            if cka_val < min_val:
                min_val = cka_val
                
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            cm[i][j] = (cm[i][j] - min_val) / (max_val - min_val)
                    
    plot_confusion_matrix(cm, labels, "confusion_matrix.png")
            
        
        
if __name__ == "__main__":
    embeddings = [
        ('coucou', 'filename')
    ]
    main(
        embeddings
    )
