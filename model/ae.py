import torch
from torch import nn
from config import opt
import torch
from torch import nn
from block import Flatten

class classifier(nn.Module):
    def __init__(self,num_classes):
        super(classifier,self).__init__()
        self.c_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            Flatten(),
            nn.Linear(128,128),
            nn.Linear(128,num_classes)
        )
    def forward(self,inp):
        return self.c_net(inp)    

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = None
        self.decoder = None
        self.build()
        
    def build(self):
        encode_cols = [2,32,64,96]
        encode_layers = []
        for i in range(3):
            self.encode_blk = nn.Sequential(
                nn.Conv3d(encode_cols[i],encode_cols[i+1],(3,3,3),(1,1,1),padding=1),
                nn.BatchNorm3d(encode_cols[i+1]),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1,2,2),(1,2,2),padding=0),
            )
            encode_layers.append(self.encode_blk)
        encode_layers.append(nn.Conv3d(96,128,(1,1,1),(1,1,1),padding=0))
        self.encoder = nn.Sequential(*encode_layers)
        
        decode_cols = [128,96,64,32]
        decode_layers = []
        for i in range(3):
            self.decode_blk = nn.Sequential(
                nn.Conv3d(decode_cols[i],decode_cols[i+1],(3,3,3),(1,1,1),padding=1),
                nn.BatchNorm3d(decode_cols[i+1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1,2,2), mode='nearest')
            )
            decode_layers.append(self.decode_blk)
        decode_layers.append(nn.Conv3d(32,2,(1,1,1),(1,1,1),padding=0))
        self.decoder = nn.Sequential(*decode_layers)
                
    def forward(self,inp,mode):
        if mode=="encoder":
            return self.encoder(inp)
        else:
            return self.decoder(inp)

if __name__=="__main__":        
    model = AutoEncoder()
    cls1 = classifier()

    inp = torch.rand(4,128,74,5,8)
    print(cls1(inp).shape)

    inp = torch.rand(4,2,74,40,64)
    print(model(inp,"encoder").shape)

    inp = torch.rand(4,128,74,5,8)
    print(model(inp,"decoder").shape)
