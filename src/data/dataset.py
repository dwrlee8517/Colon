import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.helpers import set_random_seed

# Set a global random seed here.
GLOBAL_SEED = 42
set_random_seed(GLOBAL_SEED)


def train_test_split_by_video(frames_csv, random_state=2):
    frames_data = pd.read_csv(frames_csv)

    # Group by video_id to compute per-video statistics.
    video_stats = frames_data.groupby("video_id").agg(
        num_total_frames=("frame_id", "count"),
        num_positive_frames=("is_polyps_frame", "sum")
    ).reset_index()
    video_stats["num_negative_frames"] = video_stats["num_total_frames"] - video_stats["num_positive_frames"]

    # Separate videos into two groups (mixed positive and negative vs only negative frames)
    videos_with_pos = video_stats[video_stats["num_positive_frames"] > 0]
    videos_only_neg = video_stats[video_stats["num_positive_frames"] == 0]

    # Perform an 80:20 train-test split on each group.
    train_videos_with_pos, test_videos_with_pos = train_test_split(videos_with_pos, test_size=0.2, random_state=random_state) 
    train_videos_only_neg, test_videos_only_neg = train_test_split(videos_only_neg, test_size=0.2, random_state=random_state)

    # Combine the video IDs from the two splits.
    train_video_ids = set(train_videos_with_pos["video_id"]).union(set(train_videos_only_neg["video_id"]))
    test_video_ids  = set(test_videos_with_pos["video_id"]).union(set(test_videos_only_neg["video_id"]))

    # Now, create train and test sets from the original frames_data.
    train_data = frames_data[frames_data["video_id"].isin(train_video_ids)]
    test_data  = frames_data[frames_data["video_id"].isin(test_video_ids)]

    return train_data, test_data

class RealColonDataset(Dataset):
    def __init__(self, 
                 data_dir="/radraid/dongwoolee/real_colon_data", 
                 frames_csv="/radraid2/dongwoolee/Colon/data/frames_polyps.csv", 
                 num_imgs=1000,
                 pos_ratio=0.1, 
                 min_skip_frames=10, 
                 apply_skip=True, 
                 transform=None):
        
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.frames_data = pd.read_csv(frames_csv)

        # Precompute image path for each row and add it as a column.
        self.frames_data["img_path"] = self.frames_data.apply(
            lambda row: self.get_frame(self.data_dir, row["video_id"], row["frame_id"]),
            axis=1
        )

        # Divide dataset into positive and negative labeled dataset for easier computation downstream
        self.pos_frames = self.frames_data[self.frames_data["is_polyps_frame"] == 1]
        self.neg_frames = self.frames_data[self.frames_data["is_polyps_frame"] == 0]
        
        # Depending on apply_skip flag, either enforce skipping or simply convert to list of records.
        if apply_skip:
            self.pos_samples = []
            n_total_pos = len(self.pos_frames)
            for _, group in self.pos_frames.groupby("video_id"):
                n_group = len(group)
                min_imgs = (num_imgs * n_group) // n_total_pos
                self.pos_samples += self._apply_random_skip(group, min_imgs, min_skip_frames)
            self.neg_samples = []
            n_total_neg = len(self.neg_frames)
            for _, group in self.neg_frames.groupby("video_id"):
                n_group = len(group)
                min_imgs = (num_imgs * n_group) // n_total_neg
                self.neg_samples += self._apply_random_skip(group, min_imgs, min_skip_frames)
        else:
            self.pos_samples = self.pos_frames.to_dict("records")
            self.neg_samples = self.neg_frames.to_dict("records")
        
        # Determine the required numbers.
        num_pos = int(num_imgs * pos_ratio)
        num_neg = num_imgs - num_pos
        
        # Raise error when there is less samples to sample from that the required number of images 
        if len(self.pos_samples) < num_pos:
            raise ValueError(f"Not enough positive samples after skip condition. Needed: {num_pos}, available: {len(self.pos_samples)}")
        if len(self.neg_samples) < num_neg:
            raise ValueError(f"Not enough negative samples after skip condition. Needed: {num_neg}, available: {len(self.neg_samples)}")
        
        # Randomly subsample to the required numbers.
        self.pos_samples = random.sample(self.pos_samples, num_pos)
        self.neg_samples = random.sample(self.neg_samples, num_neg)
        
        # Combine and shuffle.
        self.samples = self.pos_samples + self.neg_samples
        random.shuffle(self.samples)
    
    def _apply_random_skip(self, df, min_imgs, min_skip_frames, max_skip_frames=None):
        selected = []
        df = df.sort_values("frame_id")
        # Convert to a list of rows (or dictionaries) so we can work with them easily.
        rows = df.to_dict('records')
        n = len(rows)
        i = 0
        while i < n:
            selected.append(rows[i])
            if (n - i - 1) < min_skip_frames:
                break
            # Determine maximum skip; if max_skip_frames isn't provided, allow up to all remaining frames.
            if max_skip_frames is None:
                remaining = n - i
                dynamic_max = (n - i) // max(abs(min_imgs - len(selected)),5)
                clamped_max = min(dynamic_max, remaining // 10) if remaining >= 10 else dynamic_max
                effective_max = max(min_skip_frames, clamped_max)
            else:
                effective_max = max_skip_frames
            skip = random.randint(min_skip_frames, min(effective_max, n - i))
            i += skip
        return selected

    def __len__(self):
        return len(self.samples)
    
    def get_frame(self, basepath, video_id, frame_id):
        frames_dir = os.path.join(basepath, f"{video_id}_frames")
        img_path = os.path.join(frames_dir, f"{video_id}_{frame_id}.jpg")
        return img_path
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        img_path = row["img_path"]
        label = int(row["is_polyps_frame"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label