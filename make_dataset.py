import os
import random
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_pitch_from_mat_info_xml
from utils.utils import calc_velocites, correct_all_player_jumps_adjacent
from utils.data_utils import (
    infer_starters_from_tracking,
    sort_columns_by_original_order,
    get_valid_player_columns_in_order,
    compute_cumulative_distances
)
from utils.graph_utils import build_graph_sequence_from_condition


# .xml files in DFL -> .csv with Metrica_sports format
def convert_dfl_to_df(xy_objects, team, half, offset):
    tracking = xy_objects[half][team].xy
    ball = xy_objects[half]["Ball"].xy
    framerate = xy_objects[half][team].framerate
    n_frames, n_coords = tracking.shape
    n_players = n_coords // 2
    player_ids = np.arange(1, n_players + 1) + offset
    time = np.arange(n_frames) / framerate
    period = 1 if half == "firstHalf" else 2

    # Metrica_sports format Baseline
    df = pd.DataFrame(np.hstack([tracking, ball]), columns=[
        *[f"{team}_{i}_{ax}" for i in player_ids for ax in ["x", "y"]],
        "ball_x", "ball_y"
    ])
    df.insert(0, "Period", period)
    df.insert(1, "Time [s]", time)
    df.index.name = "Frame"
    
    # Sorting player columns
    player_cols = [col for col in df.columns if col.startswith(f"{team}_")]
    player_cols = sorted(player_cols, key=lambda x: (int(x.split("_")[1]), x.split("_")[2]))
    df = df[["Period", "Time [s]"] + player_cols + ["ball_x", "ball_y"]]
    
    return df

# Convert data to Series
def get_series(obj, key, half, offset=0, name="data"):
    data = obj[half].code.flatten()
    return pd.Series(data, index=np.arange(len(data)) + offset, name=name)


# Concat 1st half, 2nd half
def process_match(xy, possession, ballstatus):
    df_home_1 = convert_dfl_to_df(xy, "Home", "firstHalf", 0)
    player_cols = [col for col in df_home_1.columns if col.startswith("Home_") and (col.endswith("_x") or col.endswith("_y"))]
    num_players = len(player_cols) // 2
    df_away_1 = convert_dfl_to_df(xy, "Away", "firstHalf", num_players)
    df_home_2 = convert_dfl_to_df(xy, "Home", "secondHalf", 0)
    df_away_2 = convert_dfl_to_df(xy, "Away", "secondHalf", num_players)

    offset = df_home_1.index.max() + 1
    time_offset = df_home_1["Time [s]"].iloc[-1]
    for df in [df_home_2, df_away_2]:
        df.index += offset
        df["Time [s]"] += time_offset

    home = pd.concat([df_home_1, df_home_2])
    away = pd.concat([df_away_1, df_away_2])
    
    # Calculate Match time
    home["match_time"] = away["match_time"] = home["Time [s]"]
    home.loc[home["Period"] == 2, "match_time"] -= time_offset
    away.loc[away["Period"] == 2, "match_time"] -= time_offset

    # Add 'ball_active', 'ball_possession' (for team)
    active = pd.concat([
        get_series(ballstatus, "active", "firstHalf"),
        get_series(ballstatus, "active", "secondHalf", offset)
    ])
    poss = pd.concat([
        get_series(possession, "possession", "firstHalf"),
        get_series(possession, "possession", "secondHalf", offset)
    ])
    for df in [home, away]:
        df["active"] = active
        df["possession"] = poss

    return home, away

# Save DFL .xml files as .csv format
def organize_and_process(data_path, save_path):
    # Searching Folder
    files = [f for f in os.listdir(data_path) if f.endswith(".xml")]
    for f in files:
        match_id = f.split("_")[-1].split(".")[0]
        match_dir = os.path.join(data_path, match_id)
        os.makedirs(match_dir, exist_ok=True)
        shutil.move(os.path.join(data_path, f), os.path.join(match_dir, f))

    # Preprocessing for each folder
    for match_id in tqdm(os.listdir(data_path), desc = "Converting..."):
        match_dir = os.path.join(data_path, match_id)
        if not os.path.isdir(match_dir): continue

        pos, info, events = None, None, None
        for f in os.listdir(match_dir):
            if "positions_raw" in f: pos = f
            elif "matchinformation" in f: info = f
            elif "events_raw" in f: events = f

        if pos and info and events:
            xy, poss, ball, teamsheets, _ = read_position_data_xml(
                os.path.join(match_dir, pos),
                os.path.join(match_dir, info)
            )
            home, away = process_match(xy, poss, ball)

            save_match_dir = os.path.join(save_path, match_id)
            os.makedirs(save_match_dir, exist_ok=True)
            home.to_csv(os.path.join(save_match_dir, "tracking_home.csv"))
            away.to_csv(os.path.join(save_match_dir, "tracking_away.csv"))

            position_mapping = {
                "TW": 1, "LV": 2, "IVL": 3, "IVZ": 4, "IVR": 5, "RV": 6,
                "DML": 7, "DMZ": 8, "DMR": 9,
                "LM": 10, "HL": 11, "MZ": 12, "HR": 13, "RM": 14,
                "OLM": 15, "ZO": 16, "ORM": 17,
                "LA": 18, "STL": 19, "HST": 20, "STZ": 21, "STR": 22, "RA": 23
            }

            player_info_rows = []
            for team in ["Home", "Away"]:
                df_team = teamsheets[team].teamsheet.reset_index(drop=True)
                tracking_df = home if team == "Home" else away
                base_offset = 1 if team == "Home" else 21
                num_players = len(df_team)
                starters = infer_starters_from_tracking(tracking_df, team, num_players, offset=base_offset - 1)

                for i in range(len(df_team)):
                    col_name = f"{team}_{base_offset + i}"
                    position = df_team.loc[i, "position"]
                    position_num = position_mapping.get(position, 0)
                    is_starter = 1 if starters[i] else 0
                    if f"{col_name}_x" in tracking_df.columns and f"{col_name}_y" in tracking_df.columns:
                        player_xy = tracking_df[[f"{col_name}_x", f"{col_name}_y"]].dropna()
                        if not player_xy.empty:
                            start_frame = player_xy.index.min()
                            end_frame = player_xy.index.max()
                        else:
                            start_frame = end_frame = None
                    else:
                        start_frame = end_frame = None
                    player_info_rows.append({
                        "col_name": col_name,
                        "position": position_num,
                        "starter": is_starter,
                        "start_frame": start_frame,
                        "end_frame": end_frame
                    })

            df_player_info = pd.DataFrame(player_info_rows)
            df_player_info.to_csv(os.path.join(save_match_dir, "player_info.csv"), index=False)
            
            # 나중에 쓸 메타데이터 파일 복사         
            shutil.copy(os.path.join(match_dir, info), os.path.join(save_match_dir, "matchinformation.xml"))



class MultiMatchSoccerDataset(Dataset):
    def __init__(self, data_root, segment_length=250, condition_length=125, 
                 framerate=25, stride=25, use_condition_graph=False):
        self.data_root = data_root
        self.segment_length = segment_length
        self.condition_length = condition_length
        self.framerate = framerate
        self.stride = stride
        self.use_condition_graph = use_condition_graph
        self.samples = []
        self.match_data = {}
        self.load_all_matches(data_root)
    
    # Preprocess raw match data and extract valid trajectory segments
    def load_all_matches(self, data_root):
        match_ids = os.listdir(data_root)
        skip_ids = {"DFL-MAT-J03WN1"}  # Skip matches with insufficient data (red card early)
        match_ids = [m for m in match_ids if m not in skip_ids]
        for match_id in tqdm(match_ids, desc= "Data Loading..."):
            match_folder = os.path.join(data_root, match_id)
            home_path = os.path.join(match_folder, "tracking_home.csv")
            away_path = os.path.join(match_folder, "tracking_away.csv")

            home = pd.read_csv(home_path, index_col="Frame")
            away = pd.read_csv(away_path, index_col="Frame")
            # Detecting the abnormal movement
            home = correct_all_player_jumps_adjacent(home, framerate=self.framerate)
            away = correct_all_player_jumps_adjacent(away, framerate=self.framerate)
            home = calc_velocites(home)
            away = calc_velocites(away)
            home_dist = compute_cumulative_distances(home, "Home")
            away_dist = compute_cumulative_distances(away, "Away")
            home = pd.concat([home, home_dist], axis=1)
            away = pd.concat([away, away_dist], axis=1)

            common_cols = ['Period', 'Time [s]', 'match_time', 'active', 'possession']
            common = home[common_cols]
            home_only = home.drop(columns=common_cols).drop(columns=['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_speed'])
            away_only = away.drop(columns=common_cols)
            df = pd.concat([common, home_only, away_only, home_dist, away_dist], axis=1)

            self.column_order = df.columns.tolist()
            segments_info = self.extract_segments_info(df, match_id)
            if segments_info:
                self.match_data[match_id] = df
                self.samples.extend(segments_info)

    def extract_segments_info(self, df, match_id):
        segments_info = []
        num_frames = len(df)
        possession_array = df["possession"].values
        active_array = df["active"].values
        ball_x_valid = ~np.isnan(df["ball_x"].values)
        ball_y_valid = ~np.isnan(df["ball_y"].values)
        valid_mask = (active_array == 1) & ball_x_valid & ball_y_valid
        segments = []
        current_start = None
        for i in range(num_frames):
            if not valid_mask[i] or pd.isna(possession_array[i]):
                if current_start is not None:
                    segments.append((current_start, i - 1))
                    current_start = None
            else:
                if current_start is None or possession_array[i] != possession_array[current_start]:
                    if current_start is not None:
                        segments.append((current_start, i - 1))
                    current_start = i
        if current_start is not None and current_start < num_frames - self.segment_length:
            segments.append((current_start, num_frames - 1))
        for start, end in segments:
            if end - start + 1 < self.segment_length:
                continue
            i = start
            while i <= end - self.segment_length:
                segment = df.iloc[i:i + self.segment_length]
                possession_team = possession_array[i]
                # Distinguishing Attk team / Def team
                if possession_team == 1:
                    atk_prefix, def_prefix = "Home", "Away"
                elif possession_team == 2:
                    atk_prefix, def_prefix = "Away", "Home"
                else:
                    i += 1
                    continue
                atk_cols = get_valid_player_columns_in_order(segment, atk_prefix, self.column_order)
                def_cols = get_valid_player_columns_in_order(segment, def_prefix, self.column_order)
                if len(atk_cols) != 22 or len(def_cols) != 22:
                    i += 1
                    continue
                input_feats = sort_columns_by_original_order(["ball_x", "ball_y"] + atk_cols, self.column_order)
                target_feats = sort_columns_by_original_order(def_cols, self.column_order)
                segments_info.append((match_id, i, input_feats, target_feats))
                i += self.stride
        return segments_info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        match_id, start_idx, other_columns, target_columns = self.samples[idx]
        df = self.match_data[match_id]
        full_seq = df.iloc[start_idx:start_idx + self.segment_length]

        # Determine team roles (attacking or defending) based on possession
        possession_team = df.iloc[start_idx]["possession"]
        atk_prefix, def_prefix = ("Home", "Away") if possession_team == 1 else ("Away", "Home")

        # Extract 22 valid players (11 per team) from current segment
        atk_cols = get_valid_player_columns_in_order(full_seq, atk_prefix, self.column_order)
        def_cols = get_valid_player_columns_in_order(full_seq, def_prefix, self.column_order)

        if len(atk_cols) != 22 or len(def_cols) != 22:
            raise ValueError("Invalid number of valid players in segment")

        # List of player column prefixes (e.g., Home_3, Away_5)
        def get_base(col):
            return col.rsplit("_", 1)[0]  # Home_3_x -> Home_3

        atk_bases = sorted(set([get_base(c) for c in atk_cols]), key=lambda x: int(x.split('_')[1]))
        def_bases = sorted(set([get_base(c) for c in def_cols]), key=lambda x: int(x.split('_')[1]))
        player_bases = atk_bases + def_bases
        ball_feats = ["ball_x", "ball_y", "ball_vx", "ball_vy"]

        # Extract the first part of the segment as the conditioning sequence
        condition_seq = full_seq.iloc[:self.condition_length]
        target_seq = full_seq.iloc[self.condition_length:]

        # Collect feature columns
        condition_columns = set()
        for base in player_bases:
            for feat in ["x", "y", "vx", "vy", "dist"]:
                col = f"{base}_{feat}"
                if col in df.columns:
                    condition_columns.add(col)
        for col in ball_feats:
            if col in df.columns:
                condition_columns.add(col)

        # Sort columns
        condition_columns = sort_columns_by_original_order(condition_columns, self.column_order)
        condition_seq = condition_seq[condition_columns]

        # Load player metadata
        if not hasattr(self, "player_info_cache"):
            self.player_info_cache = {}

        if match_id not in self.player_info_cache:
            player_info_path = os.path.join(self.data_root, match_id, "player_info.csv")
            self.player_info_cache[match_id] = pd.read_csv(player_info_path)

        player_info = self.player_info_cache[match_id]
        player_info_map = player_info.set_index("col_name")[["position", "starter"]].to_dict("index")

        # other: Attk + ball
        # target: Def
        other_seq = target_seq[other_columns]
        target_seq = target_seq[target_columns]
        
        # Normalization
        if not hasattr(self, "pitch_cache"):
            self.pitch_cache = {}
        if match_id not in self.pitch_cache:
            info_path = os.path.join(self.data_root, match_id, "matchinformation.xml")
            pitch = read_pitch_from_mat_info_xml(info_path)
            self.pitch_cache[match_id] = (pitch.length / 2, pitch.width / 2)
        x_scale, y_scale = self.pitch_cache[match_id]
        
        # Normalization for other columns
        std_columns = ["vx", "vy", "dist", "ball_vx", "ball_vy"]
        for col in std_columns:
            if col in condition_seq.columns:
                mean = condition_seq[col].mean()
                std = condition_seq[col].std()
                condition_seq[col] = (condition_seq[col] - mean) / std

        target_seq[target_columns[0::2]] = target_seq[target_columns[0::2]] / x_scale
        target_seq[target_columns[1::2]] = target_seq[target_columns[1::2]] / y_scale

        # other_seq[other_columns[0::2]] = other_seq[other_columns[0::2]] / x_scale
        # other_seq[other_columns[1::2]] = other_seq[other_columns[1::2]] / y_scale

        condition_x_cols = [col for col in condition_seq.columns if col.endswith("_x")]
        condition_y_cols = [col for col in condition_seq.columns if col.endswith("_y")]
        condition_seq[condition_x_cols] = condition_seq[condition_x_cols] / x_scale
        condition_seq[condition_y_cols] = condition_seq[condition_y_cols] / y_scale

        # Add player's position, starter feature
        enriched_condition = []
        for i in range(len(condition_seq)):
            row = condition_seq.iloc[i].to_dict()
            enriched_row = []
            for base in player_bases:
                feats = [f"{base}_{f}" for f in ["x", "y", "vx", "vy", "dist"]]
                enriched_row.extend([float(row[f]) if f in row and pd.notna(row[f]) else 0.0 for f in feats])
                meta = player_info_map.get(base, {"position": -1, "starter": 0})
                enriched_row.append(float(meta["position"]))
                enriched_row.append(float(meta["starter"]))
            enriched_row.extend([float(row[f]) if f in row and pd.notna(row[f]) else 0.0 for f in ball_feats])
            enriched_condition.append(enriched_row)

        condition_tensor = torch.tensor(enriched_condition, dtype=torch.float32)
        other_tensor = torch.tensor(other_seq.values, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq.values, dtype=torch.float32)

        sample = {
            "match_id": match_id,
            "condition": condition_tensor,
            "other": other_tensor,
            "target": target_tensor,
            "condition_columns": [
                f"{base}_{f}" for base in player_bases for f in ["x", "y", "vx", "vy", "dist", "position", "starter"]
            ] + ball_feats,
            "other_columns": other_columns,
            "target_columns": target_columns,
            "condition_frames": list(condition_seq.index),
            "target_frames": list(target_seq.index),
            "pitch_scale": (x_scale, y_scale)
        }
        if self.use_condition_graph:
            sample["condition_graph_seq"] = build_graph_sequence_from_condition(sample, data_root=self.data_root)
        return sample


if __name__ == "__main__":
    raw_data_path = "idsse-data" # Raw Data Downloaded Path
    data_save_path = "match_data" # Saving path for preprocessed data
    organize_and_process(raw_data_path, data_save_path)
    dataset = MultiMatchSoccerDataset(data_root=data_save_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    print(len(dataset), "samples loaded.")
    sample = dataset[0]
    print("Match id:", sample["match_id"])
    print("Condition columns:", sample["condition_columns"])
    print("Condition shape:", sample["condition"].shape)
    print("Other columns:", sample["other_columns"])
    print("Other shape:", sample["other"].shape)
    print("Target columns:", sample["target_columns"])
    print("Target shape:", sample["target"].shape)
    print("Condition frames:", sample["condition_frames"])
    print("Using frames:", sample["target_frames"])
    
    from utils.data_utils import split_dataset_indices
    train_idx, test_idx, train_match_ids, test_match_ids = split_dataset_indices(dataset)

    print("\n--- Match ID Split ---")
    print(f"Train Matches ({len(train_match_ids)}): {sorted(train_match_ids)}")
    print(f"Test Matches ({len(test_match_ids)}): {sorted(test_match_ids)}")
    
    print(f"Train index: {len(train_idx)}")
    print(f"Test index: {len(test_idx)}")
    
    print("Condition:", sample["condition"])
    print("Target:", sample["target"])
    
    print("Length of condition graph seq:", len(sample["condition_graph_seq"]))
    print("Example of condition graph:", sample["condition_graph_seq"][:1] )
    
    