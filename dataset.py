import random, os, csv
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import time
from AugMix_cv2 import AugmentCV
from typing import List, Optional, Set, Tuple

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# for GRID
map = {}
num = 0
for i in range(2,36,2):
    if i==22:
        continue
    map["s"+str(i)] = num
    num+=1

# for lombard
users = [f's{i}' for i in range(2, 10)] + \
        [f's{i}' for i in range(16, 28)] + \
        [f's{i}' for i in range(44, 56)]

even_users = sorted([u for u in users if int(u[1:]) % 2 == 0], key=lambda x: int(x[1:]))
odd_users = sorted([u for u in users if int(u[1:]) % 2 == 1], key=lambda x: int(x[1:]))

even_map = {user: i for i, user in enumerate(even_users)}
odd_map = {user: i for i, user in enumerate(odd_users)}

## Dataset
class Grid(Dataset):
    def __init__(self, mode, data_path, obj, 
                 augment: bool = False, 
                 num = 250,
                 osad: bool = False,
                 osad_csv_paths: Optional[List[str]] = None,   
                 osad_cache_path: str = 'osad_anoms.txt',
                 mix_prob: bool = True,
                 ):
        super().__init__()
        assert mode in ['train', 'val', 'test', 'fake']
        self.mode = mode
        self.obj = obj 
        self.augment = augment
        self.transform = transforms.Compose([
                            transforms.Resize((50, 100)),
                            transforms.ToTensor()
                        ])
        self.target_label = map[self.obj]
        self.osad = osad
        self.osad_csv_paths = osad_csv_paths if osad_csv_paths is not None else []
        self.osad_cache_path = osad_cache_path
        
        self.osad_selected_set: Set[str] = set()  
        if self.osad and self.osad_csv_paths:
            self._ensure_osad_selection(target_label=self.target_label)
        
        self.videos: List[Tuple[str, int]] = self._load_csv_and_build_list(data_path, num=num)
        
        try:
            rows_all = self._read_csv_rows(data_path)
            self.real_pool_paths: List[str] = [vp for (vp, lb) in rows_all if lb == self.target_label]
            label_to_list = defaultdict(list)
            for vp, lb in rows_all:
                label_to_list[lb].append(vp)
            self.other_pool_paths_75: List[str] = []
            for lb, lst in label_to_list.items():
                if lb == self.target_label:
                    continue
                split_idx = int(0.75 * len(lst))
                if split_idx > 0:
                    self.other_pool_paths_75.extend(lst[:split_idx])
        except Exception:
            self.real_pool_paths = []
            self.other_pool_paths_75 = []
        
        self.mix_prob = mix_prob   
            
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_dir, raw_label = self.videos[idx]
    
        if self.mix_prob:
            p = np.random.rand()
        else:
            p = 0.5

        if self.mode == "train" and self.osad:
            if (p < 0.5) and (len(self.real_pool_paths) >= 2):
                # --- A：real-real mixup ---
                rp1, rp2 = random.sample(self.real_pool_paths, 2)
                v1 = self._load_video(rp1)
                v2 = self._load_video(rp2)
                lam = float(np.random.uniform(0.3, 0.7))
                video = (lam * v1 + (1.0 - lam) * v2).clamp(0.0, 1.0)
          
            elif (p > 0.9) and (len(self.other_pool_paths_75) > 0):
                # --- B：other identities ---
                rp = random.choice(self.other_pool_paths_75)
                video = self._load_video(rp)
               
            else:
                # --- C：fake ---
                video = self._load_video(video_dir)   # Tensor: (C, T, H, W)
                if self.augment:
                    video = self._apply_consistent_augmentation(video)  
            
            label = 1
            return video, torch.tensor(label, dtype=torch.long)

        else:
            video = self._load_video(video_dir)   # Tensor: (C, T, H, W)
            # real-0, fake-1
            if self.mode in ["test", "fake"]:
                label = 1
            elif self.mode in ["train", "val"]:
                label = 0
            else:  
                label = 0

        return video, torch.tensor(label, dtype=torch.long)
          
    def _read_csv_rows(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            return [(row[0], int(row[1])) for row in reader]
        
    def _ensure_osad_selection(self, target_label: int):
        if os.path.exists(self.osad_cache_path):
            with open(self.osad_cache_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            self.osad_selected_set = set(lines)
            return

        selected = []
        for csv_path in self.osad_csv_paths:
            rows = self._read_csv_rows(csv_path)
            neg_pool = [vp for (vp, lb) in rows if lb == target_label]
            if len(neg_pool) == 0:
                continue
            k = min(2, len(neg_pool))  # only 2 for each fake category
            chosen = random.sample(neg_pool, k=k)
            selected.extend(chosen)

        # finally 10 in total
        if len(selected) > 10:
            selected = random.sample(selected, k=10)

        self.osad_selected_set = set(selected)

        # save for fast training
        with open(self.osad_cache_path, 'w', encoding='utf-8') as f:
            for p in selected:
                f.write(str(p) + '\n')

    def _load_csv_and_build_list(self, csv_path: str, ratio: float = 0.75, num: int = 250) -> List[Tuple[str, int]]:
        # [(video_dir, raw_label), ...]
        rows = self._read_csv_rows(csv_path)
        label_to_samples = defaultdict(list)
        for video, lb in rows:
            label_to_samples[lb].append((video, lb))

        videos: List[Tuple[str, int]] = []

        if self.mode == "train" and self.osad and self.osad_selected_set:
            videos = [(vp, -1) for vp in self.osad_selected_set]
            return videos

        for lb, samples in label_to_samples.items():
            split_idx = int(ratio * len(samples))  

            if self.mode == "train":
                if lb == self.target_label:
                    pick = min(num, split_idx)
                    if pick > 0:
                        videos.extend(random.sample(samples[:split_idx], k=pick))

            elif self.mode == "val":
                if lb == self.target_label:
                    videos.extend(samples[split_idx:])

            elif self.mode == "test": # 300
                if lb != self.target_label:
                    pool = samples
                    # exclude 10 fake in training
                    if self.osad and self.osad_selected_set:
                        pool = [s for s in pool if s[0] not in self.osad_selected_set]
                    videos.extend(random.sample(pool, k=min(20, len(pool))))

            elif self.mode == "fake":
                if lb == self.target_label:
                    pool = samples
                    # exclude 10 fake in training
                    if self.osad and self.osad_selected_set:
                        pool = [s for s in pool if s[0] not in self.osad_selected_set]
                        videos.extend(pool)
                    else:
                        videos.extend(samples)

        return videos

    def _load_video(self, path, target_frames = 75): # complete path
        imgs = [self.transform(Image.open(os.path.join(path, img))) for img in sorted(os.listdir(path))] # [C, H, W] 
        
        proc = []
        for frame in imgs:
            if frame.size(0) == 1:
                frame = frame.repeat(3, 1, 1)
            # crop [C,H,W] H:5~45, W:18~82 -> [3,40,64]
            frame = frame[:, 5:45, 18:82].contiguous()
            proc.append(frame)
            if len(proc) == target_frames:
                break

        # 3) pad
        if len(proc) < target_frames:
            last = proc[-1].clone()
            proc.extend([last for _ in range(target_frames - len(proc))])
            
        # [T,C,H,W] 
        video_tensor = torch.stack(proc, dim=0)
        return video_tensor
        
    
    def _apply_consistent_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        do_flip = (np.random.rand() < 0.5)
        do_mixup = (np.random.rand() < 0.5)
       
        brightness_factor = float(np.random.uniform(0.8, 1.2))
        contrast_factor = float(np.random.uniform(0.8, 1.2))
        for t in range(T):
            frame = video[t, :, :, :]
            if do_flip:
                frame = TF.hflip(frame)
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            video[t, :, :, :] = frame
            
        if do_mixup and len(getattr(self, "real_pool_paths", [])) > 0:
            try:
                rp = random.choice(self.real_pool_paths)
                real_video = self._load_video(rp)  # (T,C,H,W)
                lam = float(np.random.uniform(0.05, 0.25))
                video = (1.0 - lam) * video + lam * real_video
                video = video.clamp(0.0, 1.0)
            except Exception:
                pass  
        
        return video  
