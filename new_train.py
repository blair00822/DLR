# 6.9 
from ae import AutoEncoder
from motion import motionNet
from block import Flatten
from new_supcon import *
from new_dataset import Grid_augmixmix, lbGrid_augmixmix
import torch
from torch import optim
from torch.utils.data import DataLoader
import os, csv, random, logging, time
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, help='epoch num when training', default=100)
    parser.add_argument('--save_interval', type=int, default=5, help='epoch interval to save checkpoint')
    parser.add_argument('--pth_root', help='saved pth file path', default='/data/users/bofanchen/model_sadth/test_new/base_ckpt') ##
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--data_path',type=str, default='/data/users/bofanchen/SA-DTH/dataset_even_new.csv', help='Path to data') ##
    parser.add_argument('--save_prefix', type=str, default='/data/users/bofanchen/model_sadth/new', help='Path to save s2 first step checkpoint') ##
    parser.add_argument('--obj', type=str, default='s2', help='choose specific user model')
    return parser.parse_args()

class Projector(nn.Module):
    def __init__(self, feat_dim=128):
        super(Projector, self).__init__()
        self.c_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            Flatten(),
            nn.Linear(128,128),
        )
        self.relu = nn.LeakyReLU() 
        self.fc = nn.Linear(128, feat_dim)
        self.batch_norm = nn.BatchNorm1d(128, affine=False)
        
    def forward(self, x):
        out = self.c_net(x)
        out1 = self.batch_norm(out)
        out2 = self.relu(out)       # out1(过BN1d), out(split)
        out2 = self.fc(out2)        # 提取128维特征
        out2 = F.normalize(out2, dim=-1) #
        return out1, out2
    
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
    
    classifier = Projector().to(device)
    
    if checkpoint_path is not None:
        state2 = torch.load(checkpoint_path, map_location = 'cpu') # trained path
        classifier.load_state_dict(state2['state_dict'], strict=False)
    else:
        classifier.load_state_dict(state1['cnet_state_dict'], strict=False)
    
    return [motion_net, ae, classifier]

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
## CUDA_VISIBLE_DEVICES=2 python new_train.py --obj s8
def train(args, model, dataset, device, save_path):
    motion_net  = model[0]
    ae          = model[1]
    check_bn_frozen(motion_net, "motion_net")
    check_bn_frozen(ae,         "autoencoder")
    
    classifier  = model[2]
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_num = len(dataset)
    
    mse_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    
    log_info = time.strftime(os.path.join('sentence_log_info', args.obj + '_NEW_log_info_%m%d_%H:%M:%S.log'))
    logger = get_logger(log_info)
    logger.info('Num of training: {}'.format(train_num))
    logger.info('====== Start (no lr scheduler)======')
    
    train_loss_list = []
    count = 0
    for epoch in range(args.epochs):
        classifier.train()
        start = time.time()
        train_loss = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for video_list, label_list in progress:
            
            all_videos = torch.cat(video_list, dim=0).to(device)  
            labels = torch.cat(label_list, dim=0).to(device) 

            pred_motion = motion_net.extract(all_videos)[0]
            encoded = ae(pred_motion.permute(0,2,1,3,4),"encoder")
            
            mid_features, features = classifier(encoded)
            loss = total_loss(mid_features, features, labels, classifier, sigma=0.05, lam=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item()) 
            progress.set_postfix(loss=loss.item())

        avg_loss = np.mean(train_loss)
        end = time.time()
        
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Time: {(end-start):.6f}")

        if len(train_loss_list) == 0 or avg_loss < min(train_loss_list):
            count = 0 
            logger.info("saving best model ...")
            torch.save({'state_dict': classifier.state_dict()}, os.path.join(save_path, 'classifier.pth'))
        else:
            count += 1
        train_loss_list.append(avg_loss)
        
        if (epoch+1) % args.save_interval == 0:
            torch.save({'state_dict': classifier.state_dict()}, os.path.join(save_path, f'classifier_epoch{epoch+1}.pth'))
        if count == 10:
            logger.info("End of Training")
            break
   
if __name__ == "__main__":
    args = arg_parse()
    set_seed(42)
    print('training user:', args.obj)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    backbone_path = os.path.join(args.pth_root, 'checkpoint.pth')
    model = get_model(backbone_path, device)
    
    # total_params = sum(p.numel() for m in model for p in m.parameters())
    # trainable_params = sum(p.numel() for m in model for p in m.parameters() if p.requires_grad)
    # ratio = trainable_params / total_params * 100

    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    # print(f"Trainable ratio: {ratio:.4f}%")
   
    dataset = Grid_augmixmix('train', data_path = args.data_path, obj=args.obj, num=750)
    
    save_path = os.path.join(args.save_prefix, args.obj, 'midDM_750_bs16_0.05new', str(args.lr))
    os.makedirs(save_path, exist_ok=True)
    
    train(args, model, dataset, device, save_path)
    
   
        
        
    
    