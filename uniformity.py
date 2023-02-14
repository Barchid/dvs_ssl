import torch
import os

def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def tolerance(x, l):
    total_distances = 0.0
    for i in range(int(l.min()), int(l.max()) + 1):
        cur_features = x[(l == i).nonzero(as_tuple=True)[0]]
        distances = torch.mm(cur_features, cur_features.T)
        mask = torch.ones((cur_features.shape[0], cur_features.shape[0])) - torch.eye(
            cur_features.shape[0]
        )
        masked_distances = distances * mask
        total_distances += masked_distances.mean()
    return total_distances.mean() / (1 + l.max() - l.min())

def get_stats(stat_file: str):
    stats = torch.load(stat_file)
    gts = stats["gts"]
    embeddings = stats["embeddings"]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    uni = uniformity(embeddings, t=2) * -1
    tol = tolerance(embeddings, gts)
    
    return uni.item(), tol.item()

def main():
    path = "/sandbox0/sami/stats"
    
    for stat_dir in os.listdir(path):
        if "dvsgesture" not in stat_dir:
            continue
        
        stat_dir_path = os.path.join(path, stat_dir)
        
        enc1 = stat_dir.split("_")[1]
        enc2 = stat_dir.split("_")[2]
        
        stat_file = os.path.join(stat_dir_path, "cka_stats.pt")
        
        uni, tol = get_stats(stat_file)
        print(f"{enc1} {enc2} UNIFORMITY={uni:.2f} TOLERANCE={tol:.2f}")

if __name__ == "__main__":
    main()