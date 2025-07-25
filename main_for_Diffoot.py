import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gc
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.Diffoot_modules import Diffoot_DenoisingNetwork
from models.Diffoot import Diffoot
from models.encoder import InteractionGraphEncoder
from dataset import CustomDataset, organize_and_process, ApplyAugmentedDataset
from utils.utils import set_everything, worker_init_fn, generator, plot_trajectories_on_pitch, log_graph_stats, calc_frechet_distance
from utils.data_utils import split_dataset_indices, compute_train_zscore_stats, custom_collate_fn
from utils.graph_utils import build_graph_sequence_from_condition

# SEED Fix
SEED = 42
set_everything(SEED)

# Save Log / Logger Setting
model_save_path = './results/logs/'
os.makedirs(model_save_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=os.path.join(model_save_path, 'train.log'),
    filemode='w'
)
logger = logging.getLogger()

# 1. Model Config & Hyperparameter Setting
csdi_config = {
    "num_steps": 1000,
    "channels": 256,
    "diffusion_embedding_dim": 256,
    "nheads": 4,
    "layers": 5,
    "side_dim": 256,
    
    "time_seq_len": 100,
    "feature_seq_len": 11,
    "compressed_dim": 32
}
hyperparams = {
    'raw_data_path': "idsse-data", # raw_data_path = "Download raw file path"
    'data_save_path': "match_data",
    'train_batch_size': 16,
    'val_batch_size': 16,
    'test_batch_size': 16,
    'num_workers': 8,
    'epochs': 30,
    'learning_rate': 1e-4,
    'num_samples': 20,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',

    'ddim_step': 50,
    'eta': 0.2,
    **csdi_config
}
num_steps = hyperparams['num_steps']
channels = hyperparams['channels']
diffusion_embedding_dim = hyperparams['diffusion_embedding_dim']
nheads = hyperparams['nheads']
layers = hyperparams['layers']
side_dim = hyperparams['side_dim']
time_seq_len = hyperparams['time_seq_len']
feature_seq_len = hyperparams['feature_seq_len']
compressed_dim = hyperparams['compressed_dim']

raw_data_path = hyperparams['raw_data_path']
data_save_path = hyperparams['data_save_path']
train_batch_size = hyperparams['train_batch_size']
val_batch_size = hyperparams['val_batch_size']
test_batch_size = hyperparams['test_batch_size']
num_workers = hyperparams['num_workers']
epochs = hyperparams['epochs']
learning_rate = hyperparams['learning_rate']
num_samples = hyperparams['num_samples']
device = hyperparams['device']
ddim_step = hyperparams['ddim_step']
eta = hyperparams['eta']
side_dim = hyperparams['side_dim']

logger.info(f"Hyperparameters: {hyperparams}")

# 2. Data Loading
print("---Data Loading---")
if not os.path.exists(data_save_path) or len(os.listdir(data_save_path)) == 0:
    organize_and_process(raw_data_path, data_save_path)
else:
    print("Skip organize_and_process")

temp_dataset = CustomDataset(data_root=data_save_path, use_graph=True)
train_idx, val_idx, test_idx = split_dataset_indices(temp_dataset, val_ratio=1/6, test_ratio=1/6, random_seed=SEED)

zscore_stats = compute_train_zscore_stats(temp_dataset, train_idx, save_path="./train_zscore_stats.pkl")
del temp_dataset
gc.collect()
dataset = CustomDataset(data_root=data_save_path, zscore_stats=zscore_stats, use_graph=True)

train_dataloader = DataLoader(
    ApplyAugmentedDataset(Subset(dataset, train_idx), use_graph=True),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
    generator=generator(SEED)
)

val_dataloader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn,
)

test_dataloader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=1,
    collate_fn=custom_collate_fn,
    worker_init_fn=worker_init_fn
)

print("---Data Load!---")
print(f"Train: {len(train_dataloader.dataset)} | Val: {len(val_dataloader.dataset)} | Test: {len(test_dataloader.dataset)}")

# 3. Model Define
# Extract node feature dimension
sample = dataset[0]
graph = build_graph_sequence_from_condition({
    "condition": sample["condition"],
    "condition_columns": sample["condition_columns"],
    "pitch_scale": sample["pitch_scale"],
    "zscore_stats": zscore_stats
}).to(device)

log_graph_stats(graph, logger, prefix="InitGraphSample")

in_dim = graph['Node'].x.size(1)

# Model Define
graph_encoder = InteractionGraphEncoder(in_dim=in_dim, hidden_dim=side_dim, out_dim=side_dim).to(device)
denoiser = Diffoot_DenoisingNetwork(csdi_config)
diff_model = Diffoot(denoiser, num_steps=num_steps).to(device)
optimizer = torch.optim.AdamW(list(diff_model.parameters()) + list(graph_encoder.parameters()), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, threshold=1e-5, min_lr=learning_rate*0.01)

logger.info(f"Device: {device}")
logger.info(f"GraphEncoder: {graph_encoder}")
logger.info(f"Denoiser (Diffoot_DenoisingNetwork): {denoiser}")
logger.info(f"Diffoot: {diff_model}")

# 4. Train
best_model_path = None
timestamp = datetime.now().strftime('%m%d')
best_val_loss = float("inf")

train_losses = []
val_losses   = []

for epoch in tqdm(range(1, epochs + 1), desc="Training...", leave=True):
    diff_model.train()
    graph_encoder.train()
    
    train_loss_v = 0
    train_noise_nll = 0
    train_loss = 0

    for batch in tqdm(train_dataloader, desc = "Batch Training...", leave=False):
        cond = batch["condition"].to(device)
        B, T_cond, _ = cond.shape
        _, T_target, _ = batch["target"].shape
        target_columns = batch["target_columns"][0]
        condition_columns = batch["condition_columns"][0]
        
        target_x_indices = []
        target_y_indices = []
        
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]
            y_col = target_columns[i + 1]
            
            if x_col in condition_columns and y_col in condition_columns:
                target_x_indices.append(condition_columns.index(x_col))
                target_y_indices.append(condition_columns.index(y_col))

        last_past_cond = cond[:, -1]

        target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
        graph_batch = batch["graph"].to(device) # HeteroData batch
        # graph → H
        H = graph_encoder(graph_batch) # [B, 256]
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        cond_info = cond_H

        # timestep (consistency)
        t = torch.randint(0, diff_model.num_steps, (target_rel.size(0),), device=device)
        
        loss_v, noise_nll = diff_model(target_rel, t=t, cond_info=cond_info)
        loss = loss_v + noise_nll * 0.001
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        
        train_loss_v += (loss_v).item()
        train_noise_nll += (noise_nll * 0.001).item()
        train_loss += loss.item()

        del cond, last_past_cond, target_rel, graph_batch, H, cond_H
        del cond_info, t, loss_v, noise_nll

    num_batches = len(train_dataloader)
    
    avg_train_loss_v = train_loss_v / num_batches
    avg_train_noise_nll = train_noise_nll / num_batches
    avg_train_loss = train_loss / num_batches


    # --- Validation ---
    diff_model.eval()
    graph_encoder.eval()
    
    val_loss_v = 0
    val_noise_nll = 0
    val_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            cond = batch["condition"].to(device)
            B, T_cond, _ = cond.shape
            _, T_target, _ = batch["target"].shape
            target_columns = batch["target_columns"][0]
            condition_columns = batch["condition_columns"][0]
            
            target_x_indices = []
            target_y_indices = []
            
            for i in range(0, len(target_columns), 2):
                x_col = target_columns[i]
                y_col = target_columns[i + 1]
                
                if x_col in condition_columns and y_col in condition_columns:
                    target_x_indices.append(condition_columns.index(x_col))
                    target_y_indices.append(condition_columns.index(y_col))
            
            last_past_cond = cond[:, -1]

            target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T, 11, 2]
            graph_batch = batch["graph"].to(device) # HeteroData batch

            # graph → H
            H = graph_encoder(graph_batch) # [B, 256]
            cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
            cond_info = cond_H
            
            t = torch.randint(0, diff_model.num_steps, (B,), device=device)
    
            loss_v, noise_nll = diff_model(target_rel, t=t, cond_info=cond_info)
            val_loss = loss_v + noise_nll * 0.001

            val_loss_v += (loss_v).item()
            val_noise_nll += (noise_nll * 0.001).item()
            val_total_loss += val_loss.item()

        del cond, last_past_cond, target_rel, graph_batch, H, cond_H
        del cond_info, t, loss_v, noise_nll

    num_batches = len(val_dataloader)

    avg_val_loss_v = val_loss_v / num_batches
    avg_val_noise_nll = val_noise_nll / num_batches
    avg_val_loss = val_total_loss / num_batches

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"[Epoch {epoch}/{epochs}] Train Loss={avg_train_loss:.6f} (Noise simple={avg_train_loss_v:.6f}, Noise NLL={avg_train_noise_nll:.6f}) | "
                f"Val Loss={avg_val_loss:.6f} (Noise simple={avg_val_loss_v:.6f}, Noise NLL={avg_val_noise_nll:.6f}) | LR={current_lr:.6e}")

    tqdm.write(f"[Epoch {epoch}]\n"
               f"[Train] Cost: {avg_train_loss:.6f} | Noise Loss: {avg_train_loss_v:.6f} | NLL Loss: {avg_train_noise_nll:.6f} | LR: {current_lr:.6f}\n"
               f"[Validation] Val Loss: {avg_val_loss:.6f} | Noise Loss: {avg_val_loss_v:.6f} | NLL Loss: {avg_val_noise_nll:.6f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        best_model_path = os.path.join(model_save_path, f'{timestamp}_best_model_epoch_{epoch}.pth')
        
        best_state_dict = {
            'diff_model': {k: v.cpu().clone() for k, v in diff_model.state_dict().items()},
            'graph_encoder': {k: v.cpu().clone() for k, v in graph_encoder.state_dict().items()},
            'zscore_stats': zscore_stats
        }
        
        torch.save({
            'epoch': epoch,
            'diff_model': diff_model.state_dict(),
            'graph_encoder': graph_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'zscore_stats': zscore_stats,
            'hyperparams': hyperparams
        }, best_model_path)
    
    torch.cuda.empty_cache()
    gc.collect()

logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

if epoch == epochs:
    for loader in (train_dataloader, val_dataloader):
        ds = loader.dataset
        ds = ds.dataset if isinstance(ds, Subset) else ds
        if hasattr(ds, "graph_cache"):
            ds.graph_cache.clear()
            
# 4-1. Plot learning_curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Train & Validation Loss, {num_steps} steps, {channels} channels,\n"
          f"{diffusion_embedding_dim} embedding dim, {nheads} heads, {layers} layers")
plt.legend()
plt.tight_layout()
plt.savefig(f'results/{timestamp}_diffusion_lr_curve.png')

plt.show()
plt.close()

# 5. Inference (Best-of-N Sampling) & Visualization
diff_model.load_state_dict({k: v.to(device) for k, v in best_state_dict['diff_model'].items()})
graph_encoder.load_state_dict({k: v.to(device) for k, v in best_state_dict['graph_encoder'].items()})

diff_model.eval()
graph_encoder.eval()

all_ades = []
all_fdes = []
all_frechet_dist = []
all_DE = []

all_min_ades = []
all_min_fdes = []
all_min_frechet = []
all_min_DE = []

px_mean = torch.tensor(zscore_stats['player_x_mean'], device=device)
px_std = torch.tensor(zscore_stats['player_x_std'], device=device)
py_mean = torch.tensor(zscore_stats['player_y_mean'], device=device)
py_std = torch.tensor(zscore_stats['player_y_std'],  device=device)

bx_mean = torch.tensor(zscore_stats['ball_x_mean'], device=device)
bx_std = torch.tensor(zscore_stats['ball_x_std'], device=device)
by_mean = torch.tensor(zscore_stats['ball_y_mean'], device=device)
by_std = torch.tensor(zscore_stats['ball_y_std'], device=device)

rel_x_mean = torch.tensor(zscore_stats['rel_x_mean'], device=device)
rel_x_std = torch.tensor(zscore_stats['rel_x_std'], device=device)
rel_y_mean = torch.tensor(zscore_stats['rel_y_mean'], device=device)
rel_y_std = torch.tensor(zscore_stats['rel_y_std'], device=device)

with torch.no_grad():        
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Test Streaming Inference", leave=True)):
        cond = batch["condition"].to(device)
        B, T_cond, _ = cond.shape
        _, T_target, _ = batch["target"].shape
        target_columns = batch["target_columns"][0]
        condition_columns = batch["condition_columns"][0]
        
        target_x_indices = []
        target_y_indices = []
        
        for i in range(0, len(target_columns), 2):
            x_col = target_columns[i]
            y_col = target_columns[i + 1]
            
            if x_col in condition_columns and y_col in condition_columns:
                target_x_indices.append(condition_columns.index(x_col))
                target_y_indices.append(condition_columns.index(y_col))
        
        last_past_cond = cond[:, -1]

        initial_pos = torch.stack([
            last_past_cond[:, target_x_indices],  # [B, 11]
            last_past_cond[:, target_y_indices]   # [B, 11]
        ], dim=-1)  # [B, 11, 2]
        
        target_abs = batch["target"].to(device).view(-1, T_target, 11, 2)  # [B, T_target, 11, 2]
        target_rel = batch["target_relative"].to(device).view(-1, T_target, 11, 2)  # [B, T_target, 11, 2]

        graph_batch = batch["graph"].to(device)

        H = graph_encoder(batch["graph"].to(device))
        cond_H = H.unsqueeze(-1).unsqueeze(-1).expand(-1, H.size(1), 11, T_target)
        cond_info = cond_H
        
        preds = diff_model.generate(shape=target_rel.shape, cond_info=cond_info, ddim_steps=ddim_step, eta=eta, num_samples=num_samples) # (B, T, 11, 2)

        # reference point denormalization
        ref_denorm = initial_pos.clone()
        ref_denorm[..., 0] = initial_pos[..., 0] * px_std + px_mean
        ref_denorm[..., 1] = initial_pos[..., 1] * py_std + py_mean

        # Target Denormalization
        target_abs_denorm = target_abs.clone()
        target_abs_denorm[..., 0] = target_abs[..., 0] * px_std + px_mean
        target_abs_denorm[..., 1] = target_abs[..., 1] * py_std + py_mean

        # Evaluation
        batch_ades_all_samples = []
        batch_fdes_all_samples = []
        batch_frechet_all_samples = []
        batch_DE_all_samples = []

        for sample_idx in range(num_samples):
            pred = preds[sample_idx]

            pred_rel_denorm = pred.clone()
            pred_rel_denorm[..., 0] = pred[..., 0] * rel_x_std + rel_x_mean
            pred_rel_denorm[..., 1] = pred[..., 1] * rel_y_std + rel_y_mean

            pred_absolute = pred_rel_denorm + ref_denorm.unsqueeze(1)
            
            # ADE & FDE
            ade = ((pred_absolute[...,:2] - target_abs_denorm[...,:2])**2).sum(-1).sqrt().mean((1,2))  # [B]
            fde = ((pred_absolute[:,-1,:,:2] - target_abs_denorm[:,-1,:,:2])**2).sum(-1).sqrt().mean(1)  # [B]
            # DE(Direction Error)
            eps = 1e-6
            overall_pred = pred_absolute[:, -1] - pred_absolute[:, 0]
            overall_gt = target_abs_denorm[:, -1] - target_abs_denorm[:, 0]
            
            norm_pred = overall_pred.norm(dim=-1, keepdim=True).clamp(min=eps)
            norm_gt = overall_gt.norm(dim=-1, keepdim=True).clamp(min=eps)
            
            u = overall_pred / norm_pred 
            v = overall_gt / norm_gt
            
            cosine = (u * v).sum(dim=-1).clamp(-1.0, 1.0)
            theta = cosine.acos()
            DE = theta.mean(dim=1)
            
            # Fréchet distance
            pred_np = pred_absolute.cpu().numpy()
            target_np = target_abs_denorm.cpu().numpy()
            B_, T, N, _ = pred_np.shape
            batch_frechet = []
            for b in range(B_):
                per_player_frechet = []
                for j in range(N):
                    pred_traj = pred_np[b, :, j, :]
                    target_traj = target_np[b, :, j, :]
                    frechet_dist = calc_frechet_distance(pred_traj, target_traj)
                    per_player_frechet.append(frechet_dist)
                batch_frechet.append(np.mean(per_player_frechet))
            
            batch_ades_all_samples.append(ade.cpu())
            batch_fdes_all_samples.append(fde.cpu())
            batch_frechet_all_samples.append(torch.tensor(batch_frechet))
            batch_DE_all_samples.append(DE.cpu())

        batch_ades_tensor = torch.stack(batch_ades_all_samples)
        batch_fdes_tensor = torch.stack(batch_fdes_all_samples)
        batch_frechet_tensor = torch.stack(batch_frechet_all_samples)
        batch_DE_tensor = torch.stack(batch_DE_all_samples)

        # Best-of-K methods
        min_ades, min_ade_indices = batch_ades_tensor.min(dim=0)  # [B]
        min_fdes, _ = batch_fdes_tensor.min(dim=0)  # [B]
        min_frechet, _ = batch_frechet_tensor.min(dim=0)  # [B]
        min_DE, _ = batch_DE_tensor.min(dim=0)  # [B]
        
        all_min_ades.extend(min_ades.tolist())
        all_min_fdes.extend(min_fdes.tolist())
        all_min_frechet.extend(min_frechet.tolist())
        all_min_DE.extend(min_DE.tolist())

        avg_ades = batch_ades_tensor.mean(dim=0)
        avg_fdes = batch_fdes_tensor.mean(dim=0)
        avg_frechet = batch_frechet_tensor.mean(dim=0)
        avg_DE = batch_DE_tensor.mean(dim=0)
        
        all_ades.extend(avg_ades.tolist())
        all_fdes.extend(avg_fdes.tolist())
        all_frechet_dist.extend(avg_frechet.tolist())
        all_DE.extend(avg_DE.tolist())
        
        print(f"[Batch {batch_idx}] "
              f"Avg - ADE={avg_ades.mean():.3f}, FDE={avg_fdes.mean():.3f}, "
              f"Frechet={avg_frechet.mean():.3f}, DE={torch.rad2deg(avg_DE.mean()):.2f}° | "
              f"Min - ADE={min_ades.mean():.3f}, FDE={min_fdes.mean():.3f}, "
              f"Frechet={min_frechet.mean():.3f}, DE={torch.rad2deg(min_DE.mean()):.2f}°")
        
        # Visualization
        timestamp = datetime.now().strftime('%m%d')
        base_dir = f"results/{timestamp}_test_trajs_best_ade"
        os.makedirs(base_dir, exist_ok=True)

        all_pred_absolutes = []
        for sample_idx in range(num_samples):
            pred = preds[sample_idx]

            pred_rel_denorm = pred.clone()
            pred_rel_denorm[..., 0] = pred[..., 0] * rel_x_std + rel_x_mean
            pred_rel_denorm[..., 1] = pred[..., 1] * rel_y_std + rel_y_mean

            pred_absolute = pred_rel_denorm + ref_denorm.unsqueeze(1)
            all_pred_absolutes.append(pred_absolute.cpu().numpy())
        
        all_pred_absolutes = np.stack(all_pred_absolutes)

        for i in range(B):
            other_cols = batch["other_columns"][i]
            target_cols = batch["target_columns"][i]

            other_seq = batch["other"][i].view(T_target, -1, 2).to(device)
            other_den = torch.zeros_like(other_seq)
            for j in range(other_seq.size(1)):
                x_col = other_cols[2 * j]
                if x_col == "ball_x":
                    x_mean, x_std = bx_mean, bx_std
                    y_mean, y_std = by_mean, by_std
                else:
                    x_mean, x_std = px_mean, px_std
                    y_mean, y_std = py_mean, py_std

                other_den[:, j, 0] = other_seq[:, j, 0] * x_std + x_mean
                other_den[:, j, 1] = other_seq[:, j, 1] * y_std + y_mean

            best_sample_idx = min_ade_indices[i].item()
            pred_traj = all_pred_absolutes[best_sample_idx, i]
            
            target_traj = target_abs_denorm[i].cpu().numpy()
            other_traj = other_den.cpu().numpy()

            best_ade = batch_ades_tensor[best_sample_idx, i].item()
            best_fde = batch_fdes_tensor[best_sample_idx, i].item()
            best_frechet = batch_frechet_tensor[best_sample_idx, i].item()
            best_DE_deg = torch.rad2deg(batch_DE_tensor[best_sample_idx, i]).item()

            defender_nums = [int(col.split('_')[1]) for col in target_cols[::2]]

            folder = os.path.join(base_dir, f"batch_{batch_idx:03d}")
            os.makedirs(folder, exist_ok=True)

            save_path = os.path.join(folder, f"sample_{i:02d}.png")
            plot_trajectories_on_pitch(
                other_traj, target_traj, pred_traj, other_columns=other_cols, 
                defenders_num=defender_nums, annotate=True, save_path=save_path
            )
        
        del preds, pred_rel_denorm, pred_absolute, target_abs_denorm, ref_denorm, ade, fde
        del cond, target_rel, target_abs, initial_pos, H, cond_H, cond_info
        torch.cuda.empty_cache()
        gc.collect()
            
# print(f"Best-of-{num_samples} Sampling:")
print(f"ADE: {np.mean(all_ades):.3f} ± {np.std(all_ades):.3f} meters")
print(f"FDE: {np.mean(all_fdes):.3f} ± {np.std(all_fdes):.3f} meters")
print(f"Fréchet: {np.mean(all_frechet_dist):.3f} ± {np.std(all_frechet_dist):.3f} meters")
print(f"DE: {np.mean(all_DE):.3f}° ± {np.std(all_DE):.3f}°")

print(f"Best-of-{num_samples} Sampling (min):")
print(f"minADE{num_samples}: {np.mean(all_min_ades):.3f} ± {np.std(all_min_ades):.3f} meters")
print(f"minFDE{num_samples}: {np.mean(all_min_fdes):.3f} ± {np.std(all_min_fdes):.3f} meters")
print(f"minFréchet{num_samples}: {np.mean(all_min_frechet):.3f} ± {np.std(all_min_frechet):.3f} meters")
print(f"minDE{num_samples}: {np.mean(all_min_DE):.3f}° ± {np.std(all_min_DE):.3f}°")