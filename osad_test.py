'''
9.17
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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
from torch.utils.data import DataLoader
from osad_train import DualSubspaceHead, classifier, get_model, arg_parse
from new_train import set_seed, get_logger

if __name__ == "__main__":
    args = arg_parse()
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    # ***************** 改这个 *****************
    for args.obj in ['s2','s6','s8', 's10','s16','s32']:
        print('Testing user:', args.obj)
        # add test datasets
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
        # 用于算 EER
        anomaly_ds = Grid(
            mode='train',
            data_path=args.data_path,
            obj=args.obj,
            augment=False,              
            osad=True,                  # 开启 OSAD
            osad_csv_paths=osad_csvs,   # 5 个 CSV
            osad_cache_path=cache_file,
            mix_prob = False
        )
        anomaly_loader = DataLoader(anomaly_ds, batch_size=10, shuffle=False, pin_memory=True)
        # print("Anomaly number for evaluating: ", len(anomaly_loader.dataset))
        
        # ==== 多个测试集 ====
        specs = {
            "pt_1vid": f"{args.obj}pt_1vid_v4.csv",
            "pt_5vid": f"{args.obj}pt_5vid_v4.csv",
            "pt_10vid": f"{args.obj}pt_10vid_v4.csv",
            "pt_15vid": f"{args.obj}pt_15vid_v4.csv",
            "ERnerf_1vid": f"{args.obj}ERnerf_1vid_v0.csv",
            "ERnerf_5vid": f"{args.obj}ERnerf_5vid_v0.csv",
            "ERnerf_10vid": f"{args.obj}ERnerf_10vid_v0.csv",
            "ERnerf_15vid": f"{args.obj}ERnerf_15vid_v0.csv",
            "mimic_1vid": f"{args.obj}mimic_1vid_v1.csv",
            "mimic_5vid": f"{args.obj}mimic_5vid_v1.csv",
            "mimic_10vid": f"{args.obj}mimic_10vid_v1.csv",
            "mimic_15vid": f"{args.obj}mimic_15vid_v1.csv",
            "talklip": f"{args.obj}_talklip.csv",
            "wav2lip": f"{args.obj}_wav2lip.csv",
            "IP_LAP": f"{args.obj}_IP_LAP.csv",
            "fom": f"{args.obj}_fom.csv",
            "simswap": f"{args.obj}_simswap.csv",
        }
        # 构建 Dataset
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

        # 依次构建 fake DataLoader
        fake_loaders = {
            name: DataLoader(ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            for name, ds in fake_datasets.items()
        }
        
        # 加入 human imposter
        anomal_hm_ds = Grid(mode='test', data_path=args.data_path, obj=args.obj, augment=False,
                        osad=False, osad_csv_paths=None, osad_cache_path=cache_file)
        anomal_hm_loader = DataLoader(anomal_hm_ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # print(len(anomal_hm_loader.dataset))
        # 加入 fake_loaders 字典
        fake_loaders["anomal_hm"] = anomal_hm_loader
        
        normal_val_ds = Grid(mode='val', data_path=args.data_path, obj=args.obj, augment=False,
                        osad=False, osad_csv_paths=None, osad_cache_path=cache_file)
        normal_val_loader = DataLoader(normal_val_ds, batch_size=args.n_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # load trained checkpoint
        E = 128  
        backbone_path = os.path.join(args.pth_root, 'checkpoint.pth')
        print('Load pre-trained from: ', backbone_path)
        epoch_num = 35
        final_path = os.path.join(args.save_prefix, args.obj, f'dr3_df2_0.005_0.8_4_50/epoch{epoch_num}.pth')
        encoder = get_model(backbone_path, device, final_path)
        head = DualSubspaceHead(E=E, hidden=64).to(device)
        ckpt = torch.load(final_path, map_location='cpu')
        head.load_state_dict(ckpt['head'], strict=True)
        
        motion_net  = encoder[0]
        ae          = encoder[1]
        c_net       = encoder[2]
        motion_net.eval(); ae.eval(); c_net.eval()
        head.eval()
        
        def eval_loader(loader, label_val):
            scores, labs = [], []
            with torch.no_grad():      
                for v, y0 in loader:
                    v = v.to(device)
                    pred_motion = motion_net.extract(v)[0]
                    encoded = ae(pred_motion.permute(0,2,1,3,4),"encoder")
                    z = c_net(encoded)      # [B, 128]
                    logits, _, _ = head(z)
                    s = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    scores.append(s)
                    labs.append(np.full_like(s, fill_value=label_val, dtype=np.float32))
            if scores:
                s = np.concatenate(scores)
                l = np.concatenate(labs)
                return s, l
            return np.array([]), np.array([])
        
        def eer_threshold(scores_real, scores_fake):
            s_all = np.concatenate([scores_real, scores_fake])
            l_all = np.concatenate([np.zeros_like(scores_real), np.ones_like(scores_fake)])
            fpr, tpr, th = roc_curve(l_all, s_all)  
            far = 1.0 - tpr                         
            frr = fpr                               
            idx = np.argmin(np.abs(far - frr))
            t_eer = th[idx]
            eer = 0.5 * (far[idx] + frr[idx])
            return float(t_eer), float(eer)
        
        with torch.no_grad():
            # normal 
            s_norm, l_norm = eval_loader(normal_val_loader, 0)
            per_metrics = {}  
            per_auc = {}
            all_s, all_l = [s_norm], [l_norm]
            
            # 用 hm imposter 算 EER
            s_hm, l_hm = eval_loader(fake_loaders["anomal_hm"], 1)
            t_eer, hm_eer = eer_threshold(s_norm, s_hm)
            # 逐 fake 计算指标
            for name, loader in fake_loaders.items():
                s_fake, l_fake = eval_loader(loader, 1)
                if s_fake.size:
                    # 单独 vs normal_val 的 AUC
                    s_pair = np.concatenate([s_norm, s_fake])
                    l_pair = np.concatenate([l_norm, l_fake])
                    try:
                        auroc = roc_auc_score(l_pair, s_pair)
                    except:
                        auroc = float('nan')
                    try:
                        aupr = average_precision_score(l_pair, s_pair) 
                    except Exception:
                        aupr = float('nan')
                        
                    # 用 hm 的 EER 阈值计算 FAR / FRR / HTER
                    far = float(np.mean(s_fake < t_eer))      # 假被当真
                    frr = float(np.mean(s_norm >= t_eer)) 
                    hter = 0.5 * (far + frr)
                    
                    eer_val = hm_eer if name == "anomal_hm" else float('nan')
                    
                    per_metrics[name] = {
                            "AUC-PR": aupr,
                            "AUC-ROC": auroc,
                            "FAR": far,
                            "FRR": frr,
                            "HTER": hter,
                            "EER": eer_val,
                            "thr_eer(hm)": t_eer,
                        }

                    all_s.append(s_fake); all_l.append(l_fake)
            
            s_all = np.concatenate(all_s); l_all = np.concatenate(all_l)
            try:
                auc = roc_auc_score(l_all, s_all)
            except:
                auc = float('nan')
                
            try:
                aupr = average_precision_score(l_all, s_all)
            except:
                aupr = float('nan')
            
            far_all = float(np.mean(s_all[l_all == 1] < t_eer))
            frr_all = float(np.mean(s_all[l_all == 0] >= t_eer))
            hter_all = 0.5 * (far_all + frr_all)
            
        # ====== 打印 ======
        per_str = " | ".join([f"{k}:AUPR={v['AUC-PR']:.4f}, AUROC={v['AUC-ROC']:.4f}, "
                            f"HTER={v['HTER']:.4f}, FAR={v['FAR']:.4f}, FRR={v['FRR']:.4f}, "
                            f"EER={v['EER']:.4f}, thr={v['thr_eer(hm)']:.4f}"
                            for k, v in per_metrics.items()]) if per_metrics else "no_fake"
        print(f"per-fake: {per_str}")
        print(f"overall : AUPR={aupr:.4f}, AUROC={auc:.4f}, "
            f"HTER={hter_all:.4f}, FAR={far_all:.4f}, FRR={frr_all:.4f}  (thr from hm={t_eer:.4f}, EER_hm={hm_eer:.4f})")

        # ====== 另存 metrics.csv ======
        metrics_rows = []
        for name, v in per_metrics.items():
            metrics_rows.append({
                "split": name,
                "AUC-PR": v["AUC-PR"],
                "AUC-ROC": v["AUC-ROC"],
                "FAR": v["FAR"],
                "FRR": v["FRR"],
                "HTER": v["HTER"],
                "EER": v["EER"],
                "thr_from_hm": v["thr_eer(hm)"],
            })
        metrics_rows.append({
            "split": "overall",
            "AUC-PR": aupr,
            "AUC-ROC": auc,
            "FAR": far_all,
            "FRR": frr_all,
            "HTER": hter_all,
            "EER": hm_eer,             # 记录 hm 的 EER 以便复现
            "thr_from_hm": t_eer,
        })

        metrics_path = "./metrics_ablation.csv"
        header = ["split", "AUC-PR", "AUC-ROC", "FAR", "FRR", "HTER", "EER", "thr_from_hm"]
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for row in metrics_rows:
                w.writerow(row)

        print(f"[saved] metrics -> {metrics_path}")  
        