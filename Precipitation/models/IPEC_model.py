import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IPECNet(nn.Module):
    def __init__(self,nc,padding_type,norm_layer,task='identification'):
        super(IPECNet, self).__init__()
        self.head1=nn.Sequential(*[ResnetBlock(nc[i],nc[i+1],padding_type,norm_layer) for i in range(len(nc)-1)])
        self.head2=nn.Sequential(*[ResnetBlock(nc[i],nc[i+1],padding_type,norm_layer) for i in range(len(nc)-1)])
        self.head3=nn.Sequential(*[ResnetBlock(nc[i],nc[i+1],padding_type,norm_layer) for i in range(len(nc)-1)])
        self.head4=nn.Sequential(*[ResnetBlock(nc[i],nc[i+1],padding_type,norm_layer) for i in range(len(nc)-1)])
        self.head5=nn.Sequential(*[ResnetBlock(nc[i],nc[i+1],padding_type,norm_layer) for i in range(len(nc)-1)])

        self.pooling=nn.MaxPool2d(2)
        self.fc1=nn.Linear(5*32*14*14,1)
        self.fc2=nn.Linear(5*32*14*14,2)
        self.task=task
        
        
    
    def forward(self,X):#X: Band3 Band4 Band6 Band3-Band4 Band4-Band6
        N=X.shape[0]
        CH1=X[:,0,:,:].view(-1,1,29,29)
        CH2=X[:,1,:,:].view(-1,1,29,29)
        CH3=X[:,2,:,:].view(-1,1,29,29)
        CH4=CH1-CH2
        CH5=CH2-CH3
        feature1=self.pooling(self.head1(CH1))
        feature2=self.pooling(self.head2(CH2))
        feature3=self.pooling(self.head3(CH3))
        feature4=self.pooling(self.head4(CH4))
        feature5=self.pooling(self.head5(CH5))

        feature=torch.cat([feature1,feature2,feature3,feature4,feature5],dim=1)
        
        if self.task=='identification':
            y=self.fc2(feature.view(N,-1))
        
        if self.task=='estimation':
            y= self.fc1(feature.view(N,-1))

        return y


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, nc_input, nc_output, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(nc_input, nc_output, padding_type, norm_layer)
        self.skip_connect = self.build_skip_connect(nc_input, nc_output,norm_layer)


    def build_conv_block(self, nc_input,nc_output, padding_type, norm_layer):
        conv_block = []
        # TODO: support padding types
        assert(padding_type == 'zero')
        p=1

        block1=[ nn.Conv2d(nc_input, nc_output, kernel_size=3, padding=p),
                norm_layer(nc_output),
                nn.ReLU(True)]

        block2=[ nn.Conv2d(nc_output, nc_output, kernel_size=3, padding=p),
                norm_layer(nc_output),
                nn.ReLU(True)]
        
        conv_block += block1
        conv_block += block2

        return nn.Sequential(*conv_block)
    
    def build_skip_connect(self,nc_input, nc_output, norm_layer):
        conv_block = [  nn.Conv2d(nc_input, nc_output, kernel_size=1),
                        norm_layer(nc_output)]
        return nn.Sequential(*conv_block)


    def forward(self, x):
        out = self.skip_connect(x) + self.conv_block(x)
        return out


if __name__ == '__main__':
    dim=3
    # model=ResnetBlock(dim,dim+1,padding_type='zero',norm_layer=nn.BatchNorm2d)
    model=IPECNet(nc=[1,16,16,32,32],padding_type='zero',norm_layer=nn.BatchNorm2d)
    a=torch.rand(10,dim,29,29)
    b=model(a)
    print(b.shape)

