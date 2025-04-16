import torch
from tqdm import tqdm
from make_dataset import MultiMatchSoccerDataset
from utils.data_utils import split_dataset_indices
from utils.graph_utils import build_graph_sequence_from_condition

def save_graph_sequences(dataset, indices, save_path):
    print(f"Saving {len(indices)} samples to {save_path} ...")
    all_graph_seqs = []
    for idx in tqdm(indices, desc="Building graph sequences"):
        sample = dataset[idx]
        graph_seq = build_graph_sequence_from_condition(sample)
        all_graph_seqs.append(graph_seq)
    torch.save(all_graph_seqs, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    data_save_path = "match_data"
    dataset = MultiMatchSoccerDataset(data_root=data_save_path)
    train_idx, val_idx, test_idx = split_dataset_indices(dataset, val_ratio=1/6, test_ratio=1/6, random_seed=42)

    save_graph_sequences(dataset, train_idx, save_path="results/train_condition_graph_seq.pt")
    save_graph_sequences(dataset, val_idx, save_path="results/val_condition_graph_seq.pt")
    save_graph_sequences(dataset, test_idx, save_path="results/test_condition_graph_seq.pt")
