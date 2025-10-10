# 融合Grid_real + Grid_augmix + Grid_mixup -> Grid_augmixmix
# add lombard
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

##Dataset
##微调 用不到optical flow 
class Grid_augmixmix(Dataset):
    def __init__(self, mode, data_path, obj, augment=True, num=750, aug_severity=5):
        super().__init__()
        assert mode in ['train', 'val', 'test', 'fake']
        self.mode = mode
        self.obj = obj 
        self.augment = augment
        self.augmentor = AugmentCV(image_size=64) 
        self.aug_severity = aug_severity

        self.transform = transforms.Compose([
                            transforms.Resize((50, 100)),
                            transforms.ToTensor()
                        ])
        
        self.videos, self.fake_pool = self.load_csv(data_path, num=num)
            
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        sample, label = self.videos[idx]
        real_video = self.load_video(sample) # [T, C, H, W]
        T, C, H, W = real_video.shape
        
        if self.mode == 'train' and self.augment:
            # ---------- AugMix ----------
            param_cache = self.augmentor._generate_param_cache(mixture_width=3, aug_severity=self.aug_severity)
            RealFrames = []
            AugFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :]) # (H, W, C)
                frame_aug = self.augmentor(real_frame, param_cache=param_cache) / 255.0
                AugFrames.append(torch.from_numpy(frame_aug).permute(2, 0, 1).float()) # (C, H, W)
                RealFrames.append(torch.from_numpy(real_frame).permute(2, 0, 1).float())
            
            # (40,64)
            RealVideo = torch.stack(RealFrames, dim = 0) # [T, C, H, W]
            AugVideo = torch.stack(AugFrames, dim = 0) # [T, C, H, W]
            # AugVideo_flows = self.get_optical_flow_tensor(AugVideo)
            
            # ---------- Mixup ----------
            lam = np.random.beta(1, 1)
            while True:
                breal_sample_path, _ = random.choice(self.videos)
                if breal_sample_path != sample:
                    break
            breal_video = self.load_video(breal_sample_path) # [T, C, H, W]
            MixupFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :])
                breal_frame = self.to_numpy_frame(breal_video[t, :, :, :])
                mixup = lam * real_frame + (1 - lam) * breal_frame
                MixupFrames.append(torch.from_numpy(mixup).permute(2, 0, 1).float())
            
            MixupVideo = torch.stack(MixupFrames, dim=0)  # (T, C, H, W)
            # MixupVideo_flows = self.get_optical_flow_tensor(MixupVideo)
            
            video_list = [RealVideo, AugVideo, MixupVideo]
            # video_flows = [self.get_optical_flow_tensor(real_video), AugVideo_flows, MixupVideo_flows]
            # # 重构 label
            label_list = [torch.tensor(1), torch.tensor(0), torch.tensor(0)]
            
            return video_list, label_list
        
        else:
            RealFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :]) # (H, W, C)
                RealFrames.append(torch.from_numpy(real_frame).permute(2, 0, 1).float())
            RealVideo = torch.stack(RealFrames, dim = 0) # [T, C, H, W]
            video_list = RealVideo
            # video_flows = self.get_optical_flow_tensor(real_video)
            label_list = torch.tensor(0 if self.mode in ['test', 'fake'] else 1)
            return video_list, label_list

    
    def to_numpy_frame(self, frame_tensor):
        frame_np = frame_tensor.permute(1, 2, 0).numpy() # (H,W,C)
        frame_np = frame_np[5:45, 18:82, :]  
        return np.repeat(frame_np, 3, axis=2) if frame_np.shape[2] == 1 else frame_np
    
    def to_numpy_gray_frame(self, frame_tensor):  # frame: (1, H, W)
        frame_np = frame_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
        gray = cv2.cvtColor((frame_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32) / 255.0 # (H, W)
    
    def load_csv(self, path, ratio = 0.75, num=250):
        with open(path) as f:
            reader = csv.reader(f)
            all_rows = [(row[0], int(row[1])) for row in reader]
        
        label_to_samples = defaultdict(list)
        for video, label in all_rows:
            label_to_samples[label].append((video, label))
        
        videos, fake_pool = [], []
        target_label = map[self.obj]
        for label, samples in label_to_samples.items():
            split_idx = int(ratio * len(samples)) ## 前750
            
            if self.mode == 'train':
                if label == target_label:
                    selected = random.sample(samples[:split_idx], k=min(num, split_idx))
                    videos.extend(selected)    
                else:
                    fake_pool.extend(samples[:split_idx]) 
            
            elif self.mode == 'val':
                if label == target_label:
                    videos.extend(samples[split_idx:]) 
            
            elif self.mode == 'test':    
                if label != target_label:
                    videos.extend(random.sample(samples, min(20, len(samples))))
            
            elif self.mode == 'fake': 
                if label == target_label: # origin fake
                    videos.extend(samples)
        
        return videos, fake_pool
       
    def load_video(self, path, target_frames = 75): #  complete path
        imgs = [self.transform(Image.open(os.path.join(path, img))) for img in sorted(os.listdir(path))] # [C, H, W] 
        if len(imgs) < target_frames:
            imgs.extend([imgs[-1].clone() for _ in range(target_frames - len(imgs))])
        
        video_tensor = torch.stack(imgs[:target_frames]) # shape: [T, C, H, W]
        return video_tensor
    
    def optical_flow(self, frame1_np, frame2_np):  # 输入已是 HWC、RGB 且裁剪过的 np.array
        # 转灰度 & 转 uint8
        gray1 = cv2.cvtColor((frame1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=3, iterations=15,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow  # (H, W, 2)
    
    def get_optical_flow_tensor(self, video_tensor): # [T, C, H, W]
        flows = []
        T = video_tensor.shape[0]
        for t in range(T - 1):
            f1 = self.to_numpy_frame(video_tensor[t, :, :, :])  # 已裁剪
            f2 = self.to_numpy_frame(video_tensor[t + 1, :, :, :])
            flow = self.optical_flow(f1, f2)  # (H, W, 2)
            flow = np.transpose(flow, (2, 0, 1))  # → (2, H, W)
            flows.append(torch.from_numpy(flow).float())
        while len(flows) < T - 1:
            flows.append(torch.zeros(2, 40, 64))
        return torch.stack(flows, dim=0)  # (T-1, 2, H, W)

## for osad
class Grid(Dataset):
    def __init__(self, mode, data_path, obj, 
                 augment: bool = False, 
                 num = 250,
                 osad: bool = False,
                 osad_csv_paths: Optional[List[str]] = None,   # 5 个 CSV 的路径列表
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
        
        self.osad_selected_set: Set[str] = set()  # 统一保存“10 个异常样本”的文件夹路径（全路径）
        if self.osad and self.osad_csv_paths:
            self._ensure_osad_selection(target_label=self.target_label)
        
        self.videos: List[Tuple[str, int]] = self._load_csv_and_build_list(data_path, num=num)
        
        # --- NEW: 构建 real 样本池，供 mixup 使用 ---
        # 若 CSV 不可读或没有该身份的样本，则回退为空列表（安全）
        try:
            rows_all = self._read_csv_rows(data_path)
            self.real_pool_paths: List[str] = [vp for (vp, lb) in rows_all if lb == self.target_label]
            # 其他身份的“前 75%”
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
    
        # 先决定是否“用两段 real 做一个伪负样本”来替换当前样本（仅 OSAD 训练用）
        if self.mix_prob:
            p = np.random.rand()
        else:
            p = 0.5

        # 异常的几种形式
        if self.mode == "train" and self.osad:
            if (p < 0.5) and (len(self.real_pool_paths) >= 2):
                # --- 分支A：同身份 real-real mixup 伪负 ---
                rp1, rp2 = random.sample(self.real_pool_paths, 2)
                v1 = self._load_video(rp1)
                v2 = self._load_video(rp2)
                lam = float(np.random.uniform(0.3, 0.7))
                video = (lam * v1 + (1.0 - lam) * v2).clamp(0.0, 1.0)
          
            elif (p > 0.9) and (len(self.other_pool_paths_75) > 0):
                # --- 分支B：抽“其他身份”的真实样本（前75%）当异常 ---
                rp = random.choice(self.other_pool_paths_75)
                video = self._load_video(rp)
                # rp1 = random.choice(self.other_pool_paths_75)
                # rp2 = random.choice(self.real_pool_paths)
                # v1 = self._load_video(rp1)
                # v2 = self._load_video(rp2)
                # lam = float(np.random.uniform(0.3, 0.7))
                # video = (lam * v1 + (1.0 - lam) * v2).clamp(0.0, 1.0)
               
            else:
                # --- 分支C：同身份的伪造样本 ---
                video = self._load_video(video_dir)   # Tensor: (C, T, H, W)
                if self.augment:
                    video = self._apply_consistent_augmentation(video)  
            
            label = 1
            return video, torch.tensor(label, dtype=torch.long)

        else:
            video = self._load_video(video_dir)   # Tensor: (C, T, H, W)
            # 非 OSAD 或不在异常集合：按场景给 0（正常）1 (异常)
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
        """
        若缓存文件存在则读取，否则：
          - 从 osad_csv_paths（长度应为 5）中每个 CSV：
              读取所有 (video_path, label)，在 label == target_label 的集合中随机选 2 个路径
          - 合计 10 个路径写入缓存
        选择结果存入 self.osad_selected_set（字符串集合）。
        """
        if os.path.exists(self.osad_cache_path):
            with open(self.osad_cache_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            self.osad_selected_set = set(lines)
            return

        selected = []
        for csv_path in self.osad_csv_paths:
            # 读取该 CSV 的所有“target_label”的样本（即作为该 obj 的异常）
            rows = self._read_csv_rows(csv_path)
            neg_pool = [vp for (vp, lb) in rows if lb == target_label]
            if len(neg_pool) == 0:
                continue
            k = min(2, len(neg_pool))  # 每个 CSV 取 2 个，不足则能取多少取多少
            chosen = random.sample(neg_pool, k=k)
            selected.extend(chosen)

        # 最终裁剪到最多 10 个
        if len(selected) > 10:
            selected = random.sample(selected, k=10)

        self.osad_selected_set = set(selected)

        # 落盘保存（每行一个路径）
        with open(self.osad_cache_path, 'w', encoding='utf-8') as f:
            for p in selected:
                f.write(str(p) + '\n')

    def _load_csv_and_build_list(self, csv_path: str, ratio: float = 0.75, num: int = 250) -> List[Tuple[str, int]]:
        """返回 [(video_dir, raw_label), ...]"""
        rows = self._read_csv_rows(csv_path)
        label_to_samples = defaultdict(list)
        for video, lb in rows:
            label_to_samples[lb].append((video, lb))

        videos: List[Tuple[str, int]] = []

        # OSAD + train：若只想要该 CSV 中参与“10 个异常”的那部分，直接返回（便于单独构建 anomaly 数据集）
        if self.mode == "train" and self.osad and self.osad_selected_set:
            videos = [(vp, -1) for vp in self.osad_selected_set]
            return videos

        for lb, samples in label_to_samples.items():
            split_idx = int(ratio * len(samples))  # 前 75% 作为 train 候选

            if self.mode == "train":
                if lb == self.target_label:
                    # 训练正常集（目标身份）从前 75% 中采样
                    pick = min(num, split_idx)
                    if pick > 0:
                        videos.extend(random.sample(samples[:split_idx], k=pick))

            elif self.mode == "val":
                if lb == self.target_label:
                    videos.extend(samples[split_idx:])

            elif self.mode == "test": # 300个
                if lb != self.target_label:
                    pool = samples
                    # OSAD：把“已选的 10 个异常样本”从测试集中剔除（避免与 fake 重复）
                    if self.osad and self.osad_selected_set:
                        pool = [s for s in pool if s[0] not in self.osad_selected_set]
                    videos.extend(random.sample(pool, k=min(20, len(pool))))

            elif self.mode == "fake":
                if lb == self.target_label:
                    pool = samples
                    # OSAD：若开启，则仅返回补集（删除本 CSV 的 10 个异常中的那部分）
                    if self.osad and self.osad_selected_set:
                        pool = [s for s in pool if s[0] not in self.osad_selected_set]
                        videos.extend(pool)
                    else:
                        # 兼容旧逻辑：返回全部（origin fake）
                        videos.extend(samples)

        return videos

    def _load_video(self, path, target_frames = 75): #  complete path
        imgs = [self.transform(Image.open(os.path.join(path, img))) for img in sorted(os.listdir(path))] # [C, H, W] 
        
        proc = []
        for frame in imgs:
            # 若是灰度图 [1,H,W]，扩成 3 通道
            if frame.size(0) == 1:
                frame = frame.repeat(3, 1, 1)
            # 直接在 [C,H,W] 上裁剪 H:5~45, W:18~82 -> [3,40,64]
            frame = frame[:, 5:45, 18:82].contiguous()
            proc.append(frame)
            if len(proc) == target_frames:
                break

        # 3) 不足就用最后一帧补齐到 target_frames
        if len(proc) < target_frames:
            last = proc[-1].clone()
            proc.extend([last for _ in range(target_frames - len(proc))])
            
        # [T,C,H,W] 
        video_tensor = torch.stack(proc, dim=0)
        return video_tensor
        
    
    def _apply_consistent_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """
        对视频的每一帧应用相同的随机变换（水平翻转 + 亮度/对比度 + mixup）。
        """
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
                real_video = self._load_video(rp)  # (T,C,H,W)，内部已对齐到 target_frames
                lam = float(np.random.uniform(0.05, 0.25))
                # 逐帧线性混合（整段一致 λ）
                video = (1.0 - lam) * video + lam * real_video
                video = video.clamp(0.0, 1.0)
            except Exception:
                pass  
        
        return video

# lombard
class lbGrid_augmixmix(Dataset):
    def __init__(self, mode, data_path, obj, augment=True, num=75, aug_severity=5, seg_len=10):
        super().__init__()
        assert mode in ['train', 'val', 'test', 'fake']
        self.SEG_LEN = seg_len  ##
        self.mode = mode
        self.obj = obj 
        self.augment = augment
        self.augmentor = AugmentCV(image_size=64) 
        self.aug_severity = aug_severity

        self.transform = transforms.Compose([
                            transforms.Resize((50, 100)),
                            transforms.ToTensor()
                        ])
        
        self.videos, self.fake_pool = self.load_csv(data_path, num=num)
        ##
        self.samples = []                               # [(folder, label, start_idx), ...]
        for folder, label in self.videos:
            n_frames = len(os.listdir(folder))
            n_seg = n_frames // self.SEG_LEN            # 可以切多少段
            for k in range(n_seg):
                self.samples.append((folder, label, k * self.SEG_LEN))
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label, st = self.samples[idx]
        real_video = self.load_video(sample, start_idx=st, target_frames=self.SEG_LEN) # [T, C, H, W]
        T, C, H, W = real_video.shape
        
        if self.mode == 'train' and self.augment:
            # ---------- AugMix ----------
            param_cache = self.augmentor._generate_param_cache(mixture_width=3, aug_severity=self.aug_severity)
            RealFrames = []
            AugFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :]) # (H, W, C)
                frame_aug = self.augmentor(real_frame, param_cache=param_cache) / 255.0
                AugFrames.append(torch.from_numpy(frame_aug).permute(2, 0, 1).float()) # (C, H, W)
                RealFrames.append(torch.from_numpy(real_frame).permute(2, 0, 1).float())
            
            # (40,64)
            RealVideo = torch.stack(RealFrames, dim = 0) # [T, C, H, W]
            AugVideo = torch.stack(AugFrames, dim = 0) # [T, C, H, W]
            # AugVideo_flows = self.get_optical_flow_tensor(AugVideo)
            
            # ---------- Mixup ----------
            lam = np.random.beta(1, 1)
            while True:
                breal_sample_path, _ = random.choice(self.videos)
                if breal_sample_path != sample:
                    break
            breal_video = self.load_video(breal_sample_path, start_idx=st, target_frames=self.SEG_LEN) # [T, C, H, W]
            MixupFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :])
                breal_frame = self.to_numpy_frame(breal_video[t, :, :, :])
                mixup = lam * real_frame + (1 - lam) * breal_frame
                MixupFrames.append(torch.from_numpy(mixup).permute(2, 0, 1).float())
            
            MixupVideo = torch.stack(MixupFrames, dim=0)  # (T, C, H, W)
            # MixupVideo_flows = self.get_optical_flow_tensor(MixupVideo)
            
            video_list = [RealVideo, AugVideo, MixupVideo]
            # video_flows = [self.get_optical_flow_tensor(real_video), AugVideo_flows, MixupVideo_flows]
            # # 重构 label
            label_list = [torch.tensor(1), torch.tensor(0), torch.tensor(0)]
            
            return video_list, label_list
        
        else:
            RealFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[t, :, :, :]) # (H, W, C)
                RealFrames.append(torch.from_numpy(real_frame).permute(2, 0, 1).float())
            RealVideo = torch.stack(RealFrames, dim = 0) # [T, C, H, W]
            video_list = RealVideo
            # video_flows = self.get_optical_flow_tensor(real_video)
            label_list = torch.tensor(0 if self.mode in ['test', 'fake'] else 1)
            return video_list, label_list

    
    def to_numpy_frame(self, frame_tensor):
        frame_np = frame_tensor.permute(1, 2, 0).numpy() # (H,W,C)
        frame_np = frame_np[5:45, 18:82, :]  
        return np.repeat(frame_np, 3, axis=2) if frame_np.shape[2] == 1 else frame_np
    
    def load_csv(self, path, ratio = 0.75, num=75):
        with open(path) as f:
            reader = csv.reader(f)
            all_rows = [(row[0], int(row[1])) for row in reader]
        
        label_to_samples = defaultdict(list)
        for video, label in all_rows:
            label_to_samples[label].append((video, label))
        
        videos, fake_pool = [], []
        target_label = even_map[self.obj]
        for label, samples in label_to_samples.items():
            split_idx = int(ratio * len(samples)) ## 前750
            
            if self.mode == 'train':
                if label == target_label:
                    selected = random.sample(samples[:split_idx], k=min(num, split_idx))
                    videos.extend(selected)    
                else:
                    fake_pool.extend(samples[:split_idx]) 
            
            elif self.mode == 'val':
                if label == target_label:
                    videos.extend(samples[split_idx:]) 
            
            elif self.mode == 'test':    
                if label != target_label:
                    videos.extend(random.sample(samples, min(20, len(samples))))
            
            elif self.mode == 'fake': 
                if label == target_label: # origin fake
                    videos.extend(samples)
        
        return videos, fake_pool
       
    def load_video(self, path, start_idx=0, target_frames = 10): #  complete path
        ##
        all_imgs = sorted(os.listdir(path))
        n_frames = len(all_imgs) // target_frames * target_frames
        # ---------- 新增：校正 start_idx ----------
        if start_idx + target_frames > n_frames:
            start_idx = max(0, n_frames - target_frames)   # 退到最后一个合法窗口
        
        slice_imgs = all_imgs[start_idx : start_idx + target_frames]
        
        imgs = [self.transform(Image.open(os.path.join(path, img))) for img in slice_imgs] # [C, H, W] 
        if len(imgs) < target_frames:
            imgs.extend([imgs[-1].clone() for _ in range(target_frames - len(imgs))])
        
        video_tensor = torch.stack(imgs[:target_frames]) # shape: [T, C, H, W]
        return video_tensor
    
    def optical_flow(self, frame1_np, frame2_np):  # 输入已是 HWC、RGB 且裁剪过的 np.array
        # 转灰度 & 转 uint8
        gray1 = cv2.cvtColor((frame1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=3, iterations=15,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow  # (H, W, 2)
    
    def get_optical_flow_tensor(self, video_tensor): # [T, C, H, W]
        flows = []
        T = video_tensor.shape[0]
        for t in range(T - 1):
            f1 = self.to_numpy_frame(video_tensor[t, :, :, :])  # 已裁剪
            f2 = self.to_numpy_frame(video_tensor[t + 1, :, :, :])
            flow = self.optical_flow(f1, f2)  # (H, W, 2)
            flow = np.transpose(flow, (2, 0, 1))  # → (2, H, W)
            flows.append(torch.from_numpy(flow).float())
        while len(flows) < T - 1:
            flows.append(torch.zeros(2, 40, 64))
        return torch.stack(flows, dim=0)  # (T-1, 2, H, W)

      
    
if __name__ == "__main__":  
    from torch.utils.data import DataLoader    
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    def show_video(video_tensor, title='Video'):
        # video_tensor: [C, T, H, W]
        T = video_tensor.shape[1]
        imgs = [video_tensor[:, t, :, :] for t in range(min(T, 8))]  # 可视前 8 帧
        grid = make_grid(torch.stack(imgs), nrow=8, normalize=True)
        plt.figure(figsize=(12, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(title)
        plt.axis('off')
        plt.savefig('VisualizeDataset2/{}.png'.format(title))

    # 1. 实例化数据集
    path = '/data/users/bofanchen/SA-DTH/dataset_even_new.csv'
    dataset = Grid_augmixmix(mode='train', data_path=path, obj='s2', aug_severity=5)
    # 2. 使用 DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True) # video_list: list of 4 tensors, 每个 tensor 是 [B, C, T, H, W]
                                                            #label_list: list of 4 tensors, 每个 tensor 是 [B]
    for video_list, _, label_list in loader:
        start = time.time()
        print(f'Video list type: {type(video_list)}')
        
        # video_list 是 list of 4 videos, 每个是 tensor [C, T, H, W]
        for i, video in enumerate(video_list):
            print(f'video {i} shape: {video.shape}')  # should be [B, C, T, H, W]

        print(f'Label list: {label_list}')
        end = time.time()
        print(f"One video takes {(end-start):.6f} seconds")
        # 假设我们刚刚从 loader 得到 video_list
        # show_video(video_list[0][1], title='Real Video')           # real (list) 中的第一个
        # show_video(video_list[1][0], title='Random Blend')         # 随机混合
        # show_video(video_list[2][0], title='Self Blend')           # 自身混合
        # show_video(video_list[1][1], title='Augmix')        # 频域过滤
        break

    
