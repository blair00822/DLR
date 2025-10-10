import torch
from torch import nn
from torch.nn import functional as F 
from block import Flatten,TimeDistributed
from config import opt

class motionNet(nn.Module):
	def __init__(self):
		super(motionNet,self).__init__()
		self.conv1 = TimeDistributed(nn.Conv2d(6,64,(3,3),stride=(1,1),padding=1))
		self.pool1 = TimeDistributed(nn.MaxPool2d(2, stride=2,padding=0))
		self.conv2 = TimeDistributed(nn.Conv2d(64,96,(3,3),stride=(1,1),padding=1))
		self.pool2 = TimeDistributed(nn.MaxPool2d(2, stride=2,padding=0))
		self.conv3 = TimeDistributed(nn.Conv2d(96,128,(3,3),stride=(1,1),padding=1))
		self.pool3 = TimeDistributed(nn.MaxPool2d(2, stride=2,padding=0))
		self.conv4 = TimeDistributed(nn.Conv2d(128,256,(3,3),stride=(1,1),padding=1))

		self.deconv3 = TimeDistributed(nn.ConvTranspose2d(256,128,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))
		self.deconv2 = TimeDistributed(nn.ConvTranspose2d(128,96,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))
		self.deconv1 = TimeDistributed(nn.ConvTranspose2d(96,64,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))

		self.pred4 = TimeDistributed(nn.Conv2d(256,2,(3,3),stride=(1,1),padding=1))
		self.flow4 = TimeDistributed(nn.ConvTranspose2d(2,2,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))

		self.pred3 = TimeDistributed(nn.Conv2d(258,2,(3,3),stride=(1,1),padding=1))
		self.flow3 = TimeDistributed(nn.ConvTranspose2d(2,2,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))

		self.pred2 = TimeDistributed(nn.Conv2d(194,2,(3,3),stride=(1,1),padding=1))
		self.flow2 = TimeDistributed(nn.ConvTranspose2d(2,2,(5,5),(2,2),padding=(2,2),output_padding=(1,1)))

		self.pred1 = TimeDistributed(nn.Conv2d(130,2,(3,3),stride=(1,1),padding=1))

	def extract(self,video):
		video=self.data_concat(video)

		self.c1 = self.conv1(video)
		self.p1 = self.pool1(self.c1)
		self.c2 = self.conv2(self.p1)
		self.p2 = self.pool2(self.c2)
		self.c3 = self.conv3(self.p2)
		self.p3 = self.pool3(self.c3)
		self.c4 = self.conv4(self.p3)

		self.d3 = self.deconv3(self.c4)
		self.d2 = self.deconv2(self.d3)
		self.d1 = self.deconv1(self.d2)

		self.p4 = self.pred4(self.c4)
		self.f4 = self.flow4(self.p4)

		self.concat3 = torch.cat([self.c3,self.d3,self.f4],2)
		self.p3 = self.pred3(self.concat3)
		self.f3 = self.flow3(self.p3)

		self.concat2 = torch.cat([self.c2,self.d2,self.f3],2)
		self.p2 = self.pred2(self.concat2)
		self.f2 = self.flow2(self.p2)

		self.concat1 = torch.cat([self.c1,self.d1,self.f2],2) 
		self.p1 = self.pred1(self.concat1)
		
		return self.p1,self.p2,self.p3,self.p4

	def data_concat(self,video):
		concat_video=torch.unsqueeze(torch.cat([video[:,0,::],video[:,1,::]],1),1)
		for i in range(1,opt.ts-1):
			image_couple=torch.unsqueeze(torch.cat([video[:,i,::],video[:,i+1,::]],1),1)
			concat_video=torch.cat([concat_video,image_couple],1)
		return concat_video

if __name__=="__main__":
	inp = torch.rand(4,75,3,40,64)
	net = motionNet()
	res = net.extract(inp)
	for elem in res:
		print(elem.shape)
