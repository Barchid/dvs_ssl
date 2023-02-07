import torch
import json
import numpy as np
import os
from argparse import ArgumentParser
from project.utils.uniform_loss import uniformity, tolerance, uniformity_orig, tolerance_orig


def main(
    name1: str,
    embeddings1: str,
    name2: str,
    embeddings2: str,
    labels_json: str
):
    dirnam = f"analys/analysis_{name1}_{name2}"
    os.makedirs(dirnam, exist_ok=True)

    anal1_file = f"{dirnam}/anal1.txt"
    anal2_file = f"{dirnam}/anal2.txt"

    # load 1
    embeddings1 = torch.load(embeddings1)

    # load 2
    embeddings2 = torch.load(embeddings2)
    
    with open(labels_json, "r") as fp:
        labels_json = json.load(fp)

    with torch.no_grad():
        # METH1: uniformity - tolerance
        print("UNIFORMITY-TOLERANCE OF M1")
        unif1, tole1 = uniformity_tolerance(embeddings1, labels_json["idx_label"])
        with open(anal1_file, "w") as report:
            report.write(f"\n\nUNIFORMITY AND TOLERANCE OF M1=[{name1}]\n")
            report.write(f"UNIFORMITY={unif1}\nTOLERANCE={tole1}\n")
            report.flush()

        # METH2: uniformity - tolerance
        print("UNIFORMITY-TOLERANCE OF M2")
        unif2, tole2 = uniformity_tolerance(embeddings2, labels_json["idx_label"])
        with open(anal2_file, "w") as report:
            report.write(f"\n\nUNIFORMITY AND TOLERANCE OF M2=[{name2}]\n")
            report.write(f"UNIFORMITY={unif2}\nTOLERANCE={tole2}\n")
            report.flush()


def uniformity_tolerance(embeddings, idx_label):
    indexes = list(range(embeddings.shape[0]))
    val_targets = torch.tensor(idx_label, dtype=torch.long)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    tol = tolerance_orig(embeddings, val_targets)
    unif = uniformity_orig(embeddings) * -1
    return unif.item(), tol.item()

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

        # uniformities[i] = uniformity(x_norm, y_norm, t=2)
        uniformities[i] = torch.linalg.norm((x_norm - y_norm)).pow(2).mul(-2).exp()

        if label_x == label_y:
            indicator = 1.0
        else:
            indicator = 0.0

        tolerances[i] = tolerance(x, y, indicator)

    # u = uniformities.mean().log()

    return uniformities.mean().log(), tolerances.mean()


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
