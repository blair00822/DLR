'''
9.17 + osad learning
'''
# model
from ae import AutoEncoder
from motion import motionNet
from block import Flatten
# dataset
from new_dataset import Grid
# other necessities
import numpy as np
import random, os, csv, logging, argparse, time
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from new_train import set_seed, get_logger

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('--n_batch_size', type=int, default=4)    
    parser.add_argument('--an_batch_size', type=int, default=4)    
    parser.add_argument('--epochs', type=int, help='epoch num when training', default=100)
    parser.add_argument('--save_interval', type=int, default=5, help='epoch interval to save checkpoint')
    parser.add_argument('--pth_root', help='saved pth file path', default='/data/users/bofanchen/model_sadth/test_new/base_ckpt') ##
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--data_path',type=str, default='/data/users/bofanchen/SA-DTH/dataset_even_new.csv', help='Path to data') ##
    parser.add_argument('--save_prefix', type=str, default='/data/users/bofanchen/model_sadth/osad', help='Path to save s2 first step checkpoint') ##
    parser.add_argument('--obj', type=str, default='s2', help='choose specific user model')
    parser.add_argument('--real_num', type=int, default=50, help='real training num') # 100
    parser.add_argument('--dr', type=int, default=3)   #   
    parser.add_argument('--df', type=int, default=2) 
    return parser.parse_args()

def check_bn_frozen(model, name="net"):
    """
    打印并返回 BatchNorm 是否全部冻结：
      - Layer 处于 eval() 状态
      - 权重 / bias / running stats 都不更新
    """
    ok = True
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.training:
                print(f"[WARN] {name}: {m} 仍在 train 模式")
                ok = False
            for p in m.parameters(recurse=False):     # weight / bias
                if p.requires_grad:
                    print(f"[WARN] {name}: {m} 的参数还在 requires_grad=True")
                    ok = False
    if ok:
        print(f"[OK]   {name}: 所有 BatchNorm 均已冻结")
    return ok

# for preparation
def orthonormalize_columns(W: torch.Tensor) -> torch.Tensor:
    # W: [E, d]
    # QR
    with torch.no_grad():
        q, r = torch.linalg.qr(W, mode='reduced')  # Q:[E,d]
        sign = torch.sign(torch.diag(r))
        sign[sign==0] = 1.0
        q = q * sign
    return q.contiguous()

def pca_basis(X: np.ndarray, d: int):
    # X: [N, E] (numpy)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S V^T
    B = Vt[:d].T  # [E,d]
    return B.astype(np.float32)
    
class DualSubspaceHead(nn.Module):
    """
    输入：z in R^E
    学习：Br in R^{E x dr}, Bf in R^{E x df}（列正交）
    输出：二分类 logits
    """
    def __init__(self, E=128, dr=3, df=2, hidden=64):
        super().__init__()
        # 可学习子空间基
        self.Br = nn.Parameter(torch.empty(E, dr))  # 真实子空间
        self.Bf = nn.Parameter(torch.empty(E, df))  # 伪造子空间
        nn.init.orthogonal_(self.Br)
        nn.init.orthogonal_(self.Bf)
        
        self.register_buffer('mu_r', torch.zeros(E))   # 真实特征均值

        # 小头：输入 concat[z, e_r, e_f] → hidden → 1
        self.fc1 = nn.Linear(E + 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        
    @torch.no_grad()
    def update_mu_r(self, z: torch.Tensor, y: torch.Tensor, momentum: float = 0.05):
        """
        z: [B,E], y: [B]  (0=real, 1=fake)
        用 batch 内的真实样本均值对 mu_r 做指数滑动更新。
        """
        mask_r = (y == 0)
        if mask_r.any():
            m = z[mask_r].mean(dim=0)          # [E]
            # EMA: mu <- (1-mom)*mu + mom*m
            self.mu_r.lerp_(m, momentum)       # 原地线性插值，数值稳定

    @staticmethod
    def proj(B, z):
        # z: [B,E], B: [E,d]
        return (z @ B) @ B.T

    def forward(self, z):
        """
        返回：
          logits: [B,1]
          e_r, e_f: 残差能量
        """
        Prz = self.proj(self.Br, z)
        Pfz = self.proj(self.Bf, z)
        e_r = ((z - Prz)**2).sum(dim=1, keepdim=True)  # [B,1]
        e_f = ((z - Pfz)**2).sum(dim=1, keepdim=True)
        x = torch.cat([z, e_r, e_f], dim=1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, e_r, e_f

    def orthogonal_regularizer(self):       # 各自正交、彼此正交”的形状
        I_r = self.Br.T @ self.Br - torch.eye(self.Br.size(1), device=self.Br.device)
        I_f = self.Bf.T @ self.Bf - torch.eye(self.Bf.size(1), device=self.Bf.device)
        cross = self.Br.T @ self.Bf
        return (I_r.pow(2).sum() + I_f.pow(2).sum() + cross.pow(2).sum())

# tailored modified
class classifier(nn.Module):
    def __init__(self, out_dim=128):
        super(classifier,self).__init__()
        self.out_dim = out_dim 
        self.c_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            Flatten(),
            nn.Linear(128, out_dim)
        )
    def forward(self,inp):
        return self.c_net(inp)  
    
def get_model(backbone_path, device, checkpoint_path=None):
    state1 = torch.load(backbone_path, map_location = 'cpu') # backbone_path
    motion_net = motionNet().to(device)
    ae         = AutoEncoder().to(device)

    motion_net.load_state_dict(state1['ffe_state_dict'])
    ae.load_state_dict(state1['ae_state_dict'])
    
    for net in (motion_net, ae):
        net.eval()                               # BN / Dropout 固定
        for p in net.parameters():
            p.requires_grad = False
    
    c_net = classifier().to(device)
    
    if checkpoint_path is not None:
        state2 = torch.load(checkpoint_path, map_location = 'cpu') # trained path
        c_net.load_state_dict(state2['c_net'], strict=True)
    else:
        c_net.load_state_dict(state1['cnet_state_dict'], strict=False)
    
    return [motion_net, ae, c_net]

# tailored modified
@torch.no_grad()
def extract_features(encoder, loader, device):
    motion_net  = encoder[0]
    ae          = encoder[1]
    c_net       = encoder[2]
    motion_net.eval(); ae.eval(); c_net.eval()
    feats, labels = [], []
    
    for video, y in loader:
        video = video.to(device)    
        pred_motion = motion_net.extract(video)[0]
        encoded = ae(pred_motion.permute(0,2,1,3,4),"encoder")
        z = c_net(encoded)  # [B, 128]
        feats.append(z.cpu())
        labels.append(y)
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

def init_subspaces_by_pca(encoder, normal_loader, anomaly_loader, E=128, dr=3, df=2, device='cuda',
                          alpha = 0.8):
    # 1) Br~
    Xr, _ = extract_features(encoder, normal_loader, device)
    Br0 = pca_basis(Xr, d=dr)  # [E,dr]

    # 2) hat{z} = (I - Pr) z_f，then Bf~
    Xf, _ = extract_features(encoder, anomaly_loader, device)
    # project to orthogonal Br0 
    Br0_t = torch.from_numpy(Br0)    # [E,dr]
    I = torch.eye(E)
    Pr0 = Br0_t @ Br0_t.T            # [E,E]
    R = (I - alpha * Pr0)                    # [E,E]
    Xf_hat = (torch.from_numpy(Xf) @ R.T).numpy()
    
    df_eff = min(df, min(Xf_hat.shape[0], Xf_hat.shape[1]))
    Bf0 = pca_basis(Xf_hat, d=df_eff)  # [E,df_eff]
    return Br0, Bf0

def cuda_stats(tag):
    torch.cuda.synchronize()
    print(f"[{tag}] alloc={torch.cuda.memory_allocated()/1e6:.1f}MB | "
          f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB | "
          f"max_alloc={torch.cuda.max_memory_allocated()/1e6:.1f}MB")
# training code
def train_dual_subspace(
    encoder, normal_loader, anomaly_loader, 
    normal_val_loader, fake_loaders: dict,
    dr=3, df=2, hidden=64, 
    lambda_rec=1.0, lambda_ortho=0.1,
    lr=5e-3, weight_decay=1e-4, epochs=100,
    device='cuda'
):
    motion_net  = encoder[0]
    ae          = encoder[1]
    check_bn_frozen(motion_net, "motion_net")
    check_bn_frozen(ae,         "autoencoder")
    
    c_net  = encoder[2]
   
    E = c_net.out_dim
    # print("E:", E)
    head = DualSubspaceHead(E=E, dr=dr, df=df, hidden=hidden).to(device)
    
    # initialize Br, Bf
    Br0, Bf0 = init_subspaces_by_pca(encoder, normal_loader, anomaly_loader, E=E, dr=dr, df=df, device=device)
    with torch.no_grad():
        head.Br.copy_(torch.from_numpy(Br0).to(device).contiguous())
        head.Bf.copy_(torch.from_numpy(Bf0).to(device).contiguous())
        head.Br.copy_(orthonormalize_columns(head.Br))
        head.Bf.copy_(orthonormalize_columns(head.Bf))

    # optimize: c_net.linear + head
    params = list(c_net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # loss func
    bce = nn.BCEWithLogitsLoss()

    def one_batch_loss(z, y):
        # z: [B,128]  y: [B] (0/1)
        logits, e_r, e_f = head(z)
        yf = y.float().unsqueeze(1)
        L_cls = bce(logits, yf)

        mask_r = (y == 0).unsqueeze(1)  # real 
        mask_f = (y == 1).unsqueeze(1)  # fake
    
        nr = mask_r.float().sum().clamp_min(1.0)
        nf = mask_f.float().sum().clamp_min(1.0)

        Prz = head.proj(head.Br, z)
        Pfz = head.proj(head.Bf, z)

        # 1) real should be better reconstructed by Br
        L1 = (((z - Prz)**2).sum(dim=1, keepdim=True) * mask_r.float()).sum() / nr
        # 2) fake should be better reconstructed by Bf
        L3 = (((z - Pfz)**2).sum(dim=1, keepdim=True) * mask_f.float()).sum() / nf
        # 3) margin loss for contrastive error
        m_push = 0.3
        L2 = (F.relu(m_push - (e_r - e_f)) * mask_f.float()).sum() / nf  # fake：e_r >= m
        L4 = (F.relu(m_push - (e_f - e_r)) * mask_r.float()).sum() / nr  # real：e_f >= m
        
        L_rec = L1 + L2 + L3 + L4
        L_ortho = head.orthogonal_regularizer()

        L = L_cls + lambda_rec * L_rec + lambda_ortho * L_ortho
        
        parts = {
            'L_total': L.detach(),
            'L_cls': L_cls.detach(),
            'L_rec': L_rec.detach(),
            'L_ortho': L_ortho.detach(),
            'L1': L1.detach(), 
            'L2': L2.detach(), 
            'L3': L3.detach(), 
            'L4': L4.detach(),
           }
        return L, logits.squeeze(1), parts

    # Concatenate a normal batch and an abnormal batch for each step
    # Create a new iterator for each epoch (normally shuffle, shuffle every round)
    normal_iter = iter(normal_loader)
    anomaly_iter = iter(anomaly_loader)

    best_auc, best_state = -1, None

    for ep in range(1, epochs+1):
        
        # torch.cuda.reset_peak_memory_stats()
        # cuda_stats(f"epoch{ep}-start")
        
        c_net.train()
        head.train()
        
        meters = defaultdict(float)
        n_batches = 0
        
        for step in range(len(normal_loader)): 
            try:
                v_n, y_n = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_loader)
                v_n, y_n = next(normal_iter)

            try:
                v_a, y_a = next(anomaly_iter)
            except StopIteration:
                anomaly_iter = iter(anomaly_loader)
                v_a, y_a = next(anomaly_iter)

            # label：normal=0，anomaly=1
            v = torch.cat([v_n, v_a], dim=0).to(device)
            y = torch.cat([torch.zeros_like(y_n), torch.ones_like(y_a)], dim=0).to(device)

            # forward for feature                                
            pred_motion = motion_net.extract(v)[0]
            encoded = ae(pred_motion.permute(0,2,1,3,4),"encoder")
            z = c_net(encoded)  # [B, 128]
            
            opt.zero_grad()
            
            L, _, parts = one_batch_loss(z, y)
            L.backward()
            opt.step()
    
            # keep orthogonal
            with torch.no_grad():
                head.Br.copy_(orthonormalize_columns(head.Br))
                head.Bf.copy_(orthonormalize_columns(head.Bf))
            
            for k, v_ in parts.items():
                meters[k] += float(v_.item())
            n_batches += 1
        
        # cuda_stats(f"epoch{ep}-after-train")
        
        # 上传 github 时删除
        # ------- Validate：在 normal_val + 各 fake 上做 AUROC -------
        # 验证：normal_val (label=0) + 每个 fake_loader (label=1)
        def eval_loader(loader, label_val):
            scores, labs = [], []
            with torch.no_grad():      
                for v, y0 in loader:
                    v = v.to(device)
                    pred_motion = motion_net.extract(v)[0]
                    encoded = ae(pred_motion.permute(0,2,1,3,4),"encoder")
                    z = c_net(encoded)  # [B, 128]
                    logits, _, _ = head(z)
                    s = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    scores.append(s)
                    labs.append(np.full_like(s, fill_value=label_val, dtype=np.float32))
            if scores:
                s = np.concatenate(scores)
                l = np.concatenate(labs)
                return s, l
            return np.array([]), np.array([])
        
        c_net.eval()
        head.eval()
        with torch.no_grad():
            s_norm, l_norm = eval_loader(normal_val_loader, 0)
            per_auc = {}
            all_s, all_l = [s_norm], [l_norm]
            for name, loader in fake_loaders.items():
                s_fake, l_fake = eval_loader(loader, 1)
                if s_fake.size:
                    # 单独 vs normal_val 的 AUC
                    s_pair = np.concatenate([s_norm, s_fake])
                    l_pair = np.concatenate([l_norm, l_fake])
                    try:
                        auc_i = roc_auc_score(l_pair, s_pair)
                    except:
                        auc_i = float('nan')
                    per_auc[name] = auc_i
                    
                    all_s.append(s_fake); all_l.append(l_fake)
            
            s_all = np.concatenate(all_s); l_all = np.concatenate(all_l)
            try:
                auc = roc_auc_score(l_all, s_all)
            except:
                auc = float('nan')
                
        # cuda_stats(f"epoch{ep}-after-val")        
        
        # 计算 epoch 均值
        avg = {k: (meters[k] / max(n_batches, 1)) for k in meters}

        if auc > best_auc:
            best_auc = auc
            best_state = {
                'c_net': c_net.state_dict(),
                'head': head.state_dict(),
                'epoch': ep, 'auc': best_auc,
            }
            
        # 日志打印：先打印 per-fake，再打印 overall
        per_str = " | ".join([f"{k}:{v:.4f}" for k,v in per_auc.items()]) if per_auc else "no_fake"
        logger.info(
            f"[Ep {ep:03d}] "
            f"total={avg.get('L_total', float('nan')):.4f}  "
            f"cls={avg.get('L_cls', float('nan')):.4f}  "
            f"rec={avg.get('L_rec', float('nan')):.4f} "
            f"(L1={avg.get('L1', float('nan')):.4f} L2={avg.get('L2', float('nan')):.4f} "
            f"L3={avg.get('L3', float('nan')):.4f} L4={avg.get('L4', float('nan')):.4f})  "
            f"ortho={avg.get('L_ortho', float('nan')):.4f}  "
            f"per-fake AUROC: {per_str}  overall={auc:.4f}  best={best_auc:.4f}"
        )
        
        if (ep % args.save_interval == 0) or (ep == args.epochs):
            save_dir = os.path.join(args.save_prefix, args.obj, f"dr{args.dr}_df{args.df}_{args.lr}_0.8_{args.n_batch_size}_{args.real_num}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({'c_net': c_net.state_dict(),
                        'head': head.state_dict(),
                        'epoch': ep, 'auc': best_auc},
                       os.path.join(save_dir, f'epoch{ep}.pth'))
        
    return best_state, head

if __name__ == "__main__":
    args = arg_parse()
    set_seed(42)
    print('training user:', args.obj)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    # encoder
    backbone_path = os.path.join(args.pth_root, 'checkpoint.pth')
    print('Load pre-trained from: ', backbone_path)

    # encoder = get_model
    encoder = get_model(backbone_path, device)
    
    # dataset
    anoms_dir = '/data/users/bofanchen/datasets/test_file/'
    osad_csvs = [
        anoms_dir + args.obj + '_fom.csv',
        anoms_dir + args.obj + '_IP_LAP.csv',
        anoms_dir + args.obj + '_simswap.csv',
        anoms_dir + args.obj + '_talklip.csv',
        anoms_dir + args.obj + '_wav2lip.csv',
    ]
    prefix = '/data/users/bofanchen/ProDet/osad_anoms_cache_txt/'
    cache_file = prefix + f'osad_anoms_cache_{args.obj}.txt'
    # ==== anomaly training set ====
    anomaly_ds = Grid(
        mode='train',
        data_path=args.data_path,
        obj=args.obj,
        augment=True,               # only for anomaly training
        osad=True,                  # turn on OSAD
        osad_csv_paths=osad_csvs,   # 
        osad_cache_path=cache_file, # already exists
    )
    # ==== normal training set ====
    normal_ds = Grid(
        mode='train',
        data_path=args.data_path,
        obj=args.obj,
        augment=False,
        num = args.real_num,        # 
        osad=False,                 # turn off OSAD
        osad_csv_paths=None,
        osad_cache_path=cache_file, # whatever
    )
    
    normal_loader = DataLoader(normal_ds, batch_size=args.n_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    neg_sampler = torch.utils.data.RandomSampler(anomaly_ds, replacement=True, num_samples=len(normal_loader)*args.an_batch_size)
    anomaly_loader = DataLoader(anomaly_ds, batch_size=args.an_batch_size, sampler=neg_sampler, num_workers=4, pin_memory=True, drop_last=True)
    
    real_num = len(normal_loader) * args.n_batch_size
    fake_num = len(anomaly_loader) * args.an_batch_size
    
    # ==== for validation ====
    specs = {
        "fake1": f"{args.obj}_talklip.csv",
        "fake2": f"{args.obj}pt_1vid_v4.csv",
        "fake3": f"{args.obj}ERnerf_1vid_v0.csv",
        "fake4": f"{args.obj}mimic_1vid_v1.csv",
    }

    # validation Dataset
    fake_datasets = {
        name: Grid(
            mode='fake',
            data_path=os.path.join(anoms_dir, suffix),
            obj=args.obj,
            augment=False,
            osad=True,
            osad_csv_paths=osad_csvs,
            osad_cache_path=cache_file,
        )
        for name, suffix in specs.items()
    }

    fake_loaders = {
        name: DataLoader(ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for name, ds in fake_datasets.items()
    }
    
    # add human imposter
    anomal_hm_ds = Grid(mode='test', data_path=args.data_path, obj=args.obj, 
                        augment=False,
                        osad=False, 
                        osad_csv_paths=None, osad_cache_path=cache_file)
    anomal_hm_loader = DataLoader(anomal_hm_ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    fake_loaders["anomal_hm"] = anomal_hm_loader

    normal_val_ds = Grid(mode='val', data_path=args.data_path, obj=args.obj, 
                        augment=False,
                        osad=False, osad_csv_paths=None, osad_cache_path=cache_file)
    normal_val_loader = DataLoader(normal_val_ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    os.makedirs('osad_log_info', exist_ok=True)
    log_info = time.strftime(os.path.join('osad_log_info', args.obj + f'_info_%m%d_%H:%M:%S.log'))
    logger = get_logger(log_info)
    logger.info('Num of training real: {}, Num of training fake: {}'.format(real_num, fake_num))
    
    # START
    best_state, head = train_dual_subspace(
        encoder=encoder,
        normal_loader=normal_loader,
        anomaly_loader=anomaly_loader,
        normal_val_loader=normal_val_loader,
        fake_loaders=fake_loaders,
        dr=args.dr, df=args.df, hidden=64,
        lambda_rec=1.0, lambda_ortho=0.1,           # ************
        lr=args.lr, weight_decay=1e-4, epochs=args.epochs,
        device=device
    )

# CUDA_VISIBLE_DEVICES=3 python osad_train.py --obj s8