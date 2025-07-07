import torch 
import torch.nn as nn
from torch.nn import functional as F

from attention import SelfAttention

class VAE_Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels , out_channels , kernel_size=3,padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels , out_channels , kernel_size=3,padding=1)
        
        if in_channels == out_channels : 
            self.residual_layer = nn.Identity()
        else : 
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
        
    # def forward(self,x:torch.Tensor) -> torch.Tensor : 
    def forward(self,x) : 
        # x(batch_size , in_channels ,  height , width)
        resid_x = x 
        x = self.groupnorm_1(x) # same image size  
        x = F.silu(x) # same image size  
        x = self.conv_1(x) # same image size  
        x = self.groupnorm_2(x) # same image size
        x = F.silu(x) # same image size
        x = self.conv_2(x) # same image size
        
        return x + self.residual_layer(resid_x)


class VAE_Attention_Block(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x(batch_size , channels , height , width)
        resid_x = x 
        x = self.groupnorm(x)

        n,c,h,w = x.shape 

        ## making the self attention between all the pixels of the image 
        # (batch_size , channels , height , width) -> (batch_size , channels , height * width)
        x = x.view((n,c,h*w))
        # (batch_size , channels , height * width) -> (batch_size , height * width , channels)
        x = x.transpose(-1 , -2)
        
        # self-attention means that Q , K, and V are the same input
        # (batch_size , height * width , channels) -> (batch_size , height * width , channels)
        x = self.attention(x)
        # (batch_size , height * width , channels) -> (batch_size , channels , height * width)
        x = x.transpose(-1 , -2) 
        
        x = x.view((n,c,h,w))
        
        x += resid_x
        return x
    
class VAE_Decoder(nn.Sequential):
    def __init__(self): 
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_Residual_Block(512,512),
            VAE_Attention_Block(512),
            VAE_Residual_Block(512,512),
            VAE_Residual_Block(512,512),
            VAE_Residual_Block(512,512),
            
            # (batch_size , 512 , height/8 , width/8)
            VAE_Residual_Block(512,512),
            
            # (batch_size , 512 , height/4 , width/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_Residual_Block(512,512),
            VAE_Residual_Block(512,512),
            VAE_Residual_Block(512,512),
            
            # (batch_size , 512 , height/2 , width/2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_Residual_Block(512,256),
            VAE_Residual_Block(256,256),
            VAE_Residual_Block(256,256),
            
            # (batch_size , 256 , height , width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            
            # (batch_size , 128 , height , width)
            VAE_Residual_Block(256,128),
            VAE_Residual_Block(128,128),
            VAE_Residual_Block(128,128),
            nn.GroupNorm(32,128),
            nn.SiLU(),
            
            # (batch_size , 3 , height , width)
            nn.Conv2d(128,3,kernel_size=3,padding=1),
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor :
        # (batch_size , 4 , height/8 , width/8)
        x /= 0.18215
        for module in self : 
             x = module(x)
        # x (batch_size , 3 , height , width)    
        return x
         