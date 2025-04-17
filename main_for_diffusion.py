import os
from joblib import Parallel, delayed
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from models.diff_modules import diff_CSDI
from models.diff_model import DiffusionTrajectoryModel
from models.encoder import InteractionGraphEncoder
from make_dataset import MultiMatchSoccerDataset, organize_and_process
from utils.utils import set_evertyhing, worker_init_fn, generator, plot_trajectories_on_pitch
from utils.data_utils import split_dataset_indices, custom_collate_fn
from utils.graph_utils import build_graph_sequence_from_condition

# 1. Hyperparameter Setting
# raw_data_path = "Download raw file path"
raw_data_path = "idsse-data"
data_save_path = "match_data"
batch_size = 64
num_workers = 8
epochs = 100
learning_rate = 1e-4
num_samples = 10
SEED = 42
n_jobs = 16

set_evertyhing(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading
print("---Data Loading---")
if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
    organize_and_process(raw_data_path, data_save_path)
else:
    print("Skip organize_and_process")

dataset = MultiMatchSoccerDataset(data_root=data_save_path)
train_idx, val_idx, test_idx = split_dataset_indices(dataset, val_ratio=1/6, test_ratio=1/6, random_seed=SEED)

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

val_dataloader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
)

test_dataloader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")

# 3. Model Define
# Extract node feature dimension
sample = dataset[0]
graph = build_graph_sequence_from_condition({
    "condition": sample["condition"].to(device),
    "condition_columns": sample["condition_columns"]
})
in_dim = graph['Node'].x.size(1)

csdi_config = {
    "num_steps": 1000,
    "channels": 64,
    "diffusion_embedding_dim": 128,
    "nheads": 4,
    "layers": 4,
    "side_dim": 128
}

graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=128, out_dim=128, heads = 2).to(device)
denoiser = diff_CSDI(csdi_config)
diff_model = DiffusionTrajectoryModel(denoiser, num_steps=csdi_config["num_steps"]).to(device)
optimizer = torch.optim.Adam(diff_model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4)

# 4. Train
best_state_dict = None
best_val_loss = float("inf")

for epoch in range(1, epochs + 1):
    diff_model.train()
    total_noise_loss = total_player_loss = total_loss = 0.0

    for batch in tqdm(train_dataloader, desc="Training"):
        # single batch 에 condition, target, graph 이 모두 들어 있음
        cond = batch["condition"].to(device)                             # [B, T, F]
        target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)  # [B, T, 11, 2]
        graph_batch = batch["graph"].to(device)                                 # HeteroData batch

        # graph → H
        H = graph_encoder(graph_batch)                                         # [B, 128]
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))                    # [B,128,11,T]

        noise_loss, player_loss = diff_model(target, cond_info=cond_H)
        loss = noise_loss + player_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_noise_loss += noise_loss.item()
        total_player_loss += player_loss.item()
        total_loss += loss.item()

    num_batches = len(train_dataloader)
    
    avg_noise_loss = total_noise_loss / num_batches
    avg_player_loss = total_player_loss / num_batches
    avg_total_loss = total_loss / num_batches

    tqdm.write(f"[Epoch {epoch}] Cost: {avg_total_loss:.6f} | "
               f"Noise Loss: {avg_noise_loss:.6f} | "
               f"Player Loss: {avg_player_loss:.6f} | "
               f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- Validation ---
    diff_model.eval()
    val_noise_loss = val_player_loss = val_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            cond = batch["condition"].to(device)
            target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)
            graph_batch = batch["graph"].to(device)

            H = graph_encoder(graph_batch)
            cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))

            noise_loss, player_loss = diff_model(target, cond_info=cond_H)
            val_noise += noise_loss.item()
            val_player += player_loss.item()
            val_total += (noise_loss+player_loss).item()

    avg_val = val_total / len(val_dataloader)
    tqdm.write(f"[Epoch {epoch}] Val Loss: {avg_val:.6f} | "
               f"Noise: {val_noise/len(val_dataloader):.6f} | "
               f"Player: {val_player/len(val_dataloader):.6f}")

    scheduler.step(avg_val)
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state_dict = diff_model.state_dict()

# 5. Inference (Best-of-N Sampling) & Visualization
diff_model.load_state_dict(best_state_dict)
diff_model.eval()
all_ade, all_fde = [], []
visualize_samples = 5
visualization_done = False

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Inference"):
        cond = batch["condition"].to(device)
        target = batch["target"].to(device).view(-1, cond.size(1), 11, 2)
        graph_batch = batch["graph"].to(device)
        B = cond.size(0)

        H = graph_encoder(graph_batch)
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, cond.size(1))

        generated = diff_model.generate(
            shape=target.shape,
            cond_info=cond_H,
            num_samples=num_samples
        )  # [N, B, T, 11, 2]
        target = target.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)

        # Denormalize
        x_scales = torch.tensor(
            [s[0] for s in batch["pitch_scale"]],
            device=device, dtype=torch.float32
        ).view(1, B, 1, 1).expand(num_samples, B, 1, 1)
        y_scales = torch.tensor(
            [s[1] for s in batch["pitch_scale"]],
            device=device, dtype=torch.float32
        ).view(1, B, 1, 1).expand(num_samples, B, 1, 1)

        generated[..., 0] *= x_scales
        generated[..., 1] *= y_scales
        target[..., 0] *= x_scales
        target[..., 1] *= y_scales

        ade = ((generated - target) ** 2).sum(-1).sqrt().mean(2).mean(2)  # [N, B]
        best_idx = ade.argmin(dim=0)                                    # [B]
        best_pred = generated[best_idx, torch.arange(B)]                # [B, T, 11, 2]
        best_target = target[0, torch.arange(B)]                        # [B, T, 11, 2]

        ade_final = ((best_pred - best_target) ** 2).sum(-1).sqrt() \
                        .mean(1).mean(0)                               # [B]
        fde_final = ((best_pred[:, -1] - best_target[:, -1]) ** 2)\
                        .sum(-1).sqrt()                               # [B]

        all_ade.extend(ade_final.cpu().numpy())
        all_fde.extend(fde_final.cpu().numpy())

        if not visualization_done:
            os.makedirs("results", exist_ok=True)
            for i in range(min(B, visualize_samples)):
                others = batch["other"][i].view(-1, 12, 2).cpu()
                target_vis = best_target[i].cpu()
                pred_vis = best_pred[i].cpu()
                pitch_scale = batch["pitch_scale"][i]
                save_path = f"results/Diff_sample_{i:02d}.png"
                plot_trajectories_on_pitch(
                    others, target_vis, pred_vis, pitch_scale,
                    save_path=save_path
                )
            visualization_done = True

avg_ade = np.mean(all_ade)
avg_fde = np.mean(all_fde)
print(f"[Inference - Best of {num_samples}] ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")
