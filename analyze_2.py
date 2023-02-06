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
    dirnam = f"analys/analysis_{name1}_{name2}"
    os.makedirs(dirnam, exist_ok=True)

    overlap_file = f"{dirnam}/overlap.txt"
    anal1_file = f"{dirnam}/anal1.txt"
    anal2_file = f"{dirnam}/anal2.txt"
    
    # load 1
    embeddings1 = torch.load(embeddings1)
    with open(predictions1, "r") as fp:
        predictions1 = json.load(fp)

    # load 2
    embeddings2 = torch.load(embeddings2)
    with open(predictions2, "r") as fp:
        predictions2 = json.load(fp)

    total = len(predictions2["good"]) + len(predictions2["bad"])

    # PREDICTION OVERLAP
    m1_only = 0
    m2_only = 0
    both = 0
    neither = 0

    print("OVERLAP COMPUTATION")
    for idx in range(total):
        in_m1 = idx in predictions1["good"]
        out_m1 = not in_m1

        in_m2 = idx in predictions2["good"]
        out_m2 = not in_m2

        if in_m1 and in_m2:
            both += 1

        elif in_m1 and out_m2:
            m1_only += 1
        elif out_m1 and in_m2:
            m2_only += 1
        else:
            neither += 1

    with open(overlap_file, "w") as report:
        report.write(f"\n\nLINEAR OVERLAP OF M1=[{name1}] AND M2=[{name2}]\n")
        report.write(
            f"BOTH={both}\nNEITHER={neither}\n[M1]={m1_only}\n[M2]={m2_only}\n"
        )
        report.flush()

    with torch.no_grad():
        # METH1: uniformity - tolerance
        print("UNIFORMITY-TOLERANCE OF M1")
        unif1, tole1 = uniformity_tolerance(embeddings1, predictions1["idx_label"])
        with open(anal1_file, "w") as report:
            report.write(f"\n\nUNIFORMITY AND TOLERANCE OF M1=[{name1}]\n")
            report.write(f"UNIFORMITY={unif1}\nTOLERANCE={tole1}\n")
            report.flush()

        # METH2: uniformity - tolerance
        print("UNIFORMITY-TOLERANCE OF M2")
        unif2, tole2 = uniformity_tolerance(embeddings2, predictions2["idx_label"])
        with open(anal2_file, "w") as report:
            report.write(f"\n\nUNIFORMITY AND TOLERANCE OF M2=[{name2}]\n")
            report.write(f"UNIFORMITY={unif2}\nTOLERANCE={tole2}\n")
            report.flush()


def uniformity_tolerance(embeddings, idx_label):
    indexes = list(range(embeddings.shape[0]))

    pairs = [(a, b) for idx, a in enumerate(indexes) for b in indexes[idx + 1 :]]
    uniformities = torch.zeros(len(pairs))
    tolerances = torch.zeros(len(pairs))

    for (i, j) in pairs:
        x = embeddings[i]
        x_norm = x / torch.linalg.norm(x)  # normalize
        label_x = idx_label[i]

        y = embeddings[j]
        y_norm = y / torch.linalg.norm(y)  # normalize
        label_y = idx_label[j]

        uniformities[i] = uniformity(x_norm, y_norm, t=2)

        if label_x == label_y:
            indicator = 1.0
        else:
            indicator = 0.0

        tolerances[i] = tolerance(x, y, indicator)

    return uniformities.mean(), tolerances.mean()


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
