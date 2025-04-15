import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from make_dataset import MultiMatchSoccerDataset
from encoder import GraphAutoencoder
from tqdm import tqdm
import os

from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_evertyhing, worker_init_fn, generator
from utils.data_utils import split_dataset_indices, custom_collate_fn

raw_data_path = "idsse-data"
data_save_path = "match_data"
batch_size = 64
num_workers = 8
epochs = 100
learning_rate = 1e-4
SEED = 42

set_evertyhing(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dataset
print("---Data Loading---")
if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
    organize_and_process(raw_data_path, data_save_path)
else:
    print("Skip organize_and_process")

dataset = MultiMatchSoccerDataset(data_root=data_save_path, use_condition_graph=True)
train_idx, _, _ = split_dataset_indices(dataset, val_ratio=1/6, test_ratio=1/6, random_seed=SEED)

train_dataloader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
    generator=generator(SEED)
)

# 2. Input feature dimension extraction
sample = dataset[0]
first_frame = sample["condition_graph_seq"][0]
in_dim_dict = {
    node_type: first_frame[node_type].x.shape[-1]
    for node_type in first_frame.node_types
}

# 3. 모델 초기화
model = GraphAutoencoder(
    in_dim_dict=in_dim_dict,
    hidden_dim=64,
    temporal_hidden_dim=64,
    out_dim=128  # 최종 H 크기
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for batch in tqdm(train_dataloader, desc="Check"):
    print("Batch keys:", batch.keys())

    condition_seq = [graph.to(device) for graph in batch["condition_graph_seq"]]
    print("Type of condition_seq:", type(condition_seq))
    print("Length of condition_seq (frames):", len(condition_seq))

    first_frame = condition_seq[0]
    print("First frame type:", type(first_frame))

    print(" - Node types:", first_frame.node_types)
    print(" - Edge types:", first_frame.edge_types)

    for node_type in first_frame.node_types:
        print(f"   {node_type} - x :", first_frame[node_type].x)

    break



# 4. 학습 루프
print("Training Autoencoder...")
for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        graph_seq = batch["condition_graph_seq"]
        graph_seq = [g.to(device) for g in graph_seq]

        last_frame = graph_seq[-1]
        last_frame = last_frame.to(device)

        optimizer.zero_grad()
        H = model(graph_seq)
        loss = model.decode_from_H(H, last_frame)  # self-supervised loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    tqdm.write(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "pretrained_graph_autoencoder.pt")
