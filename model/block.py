import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x=x.reshape(-1,shape)
        return x

class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1): # 时间通常是第二个维度
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            #only support tdim=1
            inp_shape = args[0].shape # args[0] 就是 input_data
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.reshape(bs*seq_len, *x.shape[2:]) for x in args], **kwargs) # forward 方法会将输入数据的时间步维度展平，然后将数据传递给 module
            out_shape = out.shape
            return out.reshape(bs, seq_len,*out_shape[1:]) # 然后再恢复原始的形状

    def low_mem_forward(self, *args, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)
        
    def __repr__(self):
        return f'TimeDistributed({self.module})'
