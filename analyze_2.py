import torch
import json
import numpy as np
import os
from argparse import ArgumentParser
from project.utils.uniform_loss import uniformity, tolerance

def main(
    name1: str,
    embeddings1: str,
    predictions1: str,
    name2: str,
    embeddings2: str,
    predictions2: str,
):
    filename = f'analysis_{name1}_{name2}.txt'
    
    # load 1
    embeddings1 = torch.load(embeddings1)
    with open(predictions1, "r") as fp:
        predictions1 = json.load()

    # load 2
    embeddings2 = torch.load(embeddings2)
    with open(predictions2, "r") as fp:
        predictions2 = json.load()

    total = len(predictions2["good"]) + len(predictions2["bad"])
    indicators = []


    # PREDICTION OVERLAP
    m1_only = 0
    m2_only = 0
    both = 0
    neither = 0

    for idx in range(total):
        in_m1 = idx in predictions1["good"]
        out_m1 = not in_m1

        in_m2 = idx in predictions2["good"]
        out_m2 = not in_m2

        if in_m1 and in_m2:
            both += 1
            indicators.append(idx) # for tolerance computation
            
        elif in_m1 and out_m2:
            m1_only += 1
        elif out_m1 and in_m2:
            m2_only += 1
        else:
            neither += 1
    
    with open(filename, 'w') as report:
        report.write(f'\n\nLINEAR OVERLAP OF M1=[{name1}] AND M2=[{name2}]\n')
        report.write(f"BOTH={both}\nNEITHER={neither}\n[M1]={m1_only}\n[M2]={m2_only}\n")
        report.flush()
        
    with torch.no_grad():
        # uniformity - tolerance
        uniformities = torch.zeros(embeddings1.shape[0])
        tolerances = torch.zeros(embeddings1.shape[0])
        
        for i in range(embeddings1.shape[0]):
            x = embeddings1[i]
            x_norm = x / torch.linalg.norm(x) # normalize
            
            y = embeddings2[i]
            y_norm = y / torch.linalg.norm(y) # normalize
            
            uniformities[i] = uniformity(x_norm, y_norm, t=2)
            
            if i in indicators:
                indicator = 1.
            else:
                indicator = 0.
                
            tolerances[i] = tolerance(x, y, indicator)
            
        with open(filename, 'a') as report:
            report.write(f'\n\nUNIFORMITY=[{name1}] AND M2=[{name2}]\n')
            report.write(f"BOTH={both}\nNEITHER={neither}\n[M1]={m1_only}\n[M2]={m2_only}\n")
            report.flush()
            
        
        
    
        


if __name__ == "__main__":
    parser = ArgumentParser("coucou")
    parser.add_argument("--name1", required=True, type=str)
    parser.add_argument("--embeddings1", required=True, type=str)
    parser.add_argument("--predictions1", default=None, type=str)

    parser.add_argument("--name2", required=True, type=str)
    parser.add_argument("--embeddings2", required=True, type=str)
    parser.add_argument("--predictions2", default=None, type=str)
    args = parser.parse_args()

    main(
        name1=args.name1,
        embeddings1=args.embeddings1,
        predictions1=args.predictions1,
        name2=args.name2,
        embeddings2=args.embeddings2,
        predictions2=args.predictions2,
    )
