import torch
import torch.nn as nn 
from torch.nn import functional as F
from decoder import VAE_Attention_Block , VAE_Residual_Block

#### building the encoder ###
class VAE_Encoder(nn.Sequential) : 
    def __init__(self):
        super().__init__(
            # (batch_size , 3 channels , height , width) -> (batch_size , 128 channels , height , width)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            # (batch_size , 128 channels , height , width) -> (batch_size , 128 channels , height , width)
            
            VAE_Residual_Block(128,128),# combination of convolution and normalization  
            VAE_Residual_Block(128,128),# combination of convolution and normalization  
            
            # (batch_size , 128 channels , height , width) -> (batch_size , 128 channels , height/2 , width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0), # stride=2 -> skip 2 pixels when moving the kernel
            
            # (batch_size , 128 channels , height/2 , width/2) -> (batch_size , 256 channels , height/2 , width/2)
            VAE_Residual_Block(128,256),# combination of convolution and normalization  
            # (batch_size , 256 channels , height/2 , width/2) -> (batch_size , 256 channels , height/2 , width/2) 
            # Increasing the features (channels) means that each pixel contains more information and the size of the image is decreasing 
            VAE_Residual_Block(256,256),# combination of convolution and normalization  
            
            # (batch_size , 256 channels , height/2 , width/2) -> (batch_size , 256 channels , height/4 , width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            
            # (batch_size , 256 channels , height/4 , width/4) -> (batch_size , 512 channels , height/4 , width/4) 
            VAE_Residual_Block(256,512),# combination of convolution and normalization
            # (batch_size , 512 channels , height/4 , width/4) -> (batch_size , 512 channels , height/4 , width/4)
            VAE_Residual_Block(512,512),# combination of convolution and normalization
            
            # (batch_size , 512 channels , height/4 , width/4) -> (batch_size , 512 channels , height/8 , width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            
             # (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 512 channels , height/8 , width/8)
            VAE_Residual_Block(512,512),# combination of convolution and normalization
            VAE_Residual_Block(512,512),# combination of convolution and normalization
            VAE_Residual_Block(512,512),# combination of convolution and normalization
            
            # (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 512 channels , height/8 , width/8)
            VAE_Attention_Block(512), # attention between pixels : a way to relate pixels to each other 
            # attention is more general than convolution : in attention even the last pixel will be related to the first one
            
            # (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 512 channels , height/8 , width/8)
            VAE_Residual_Block(512,512),# combination of convolution and normalization
            
            # Normalization (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 512 channels , height/8 , width/8)
            nn.GroupNorm(32,512),
            
            # (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 512 channels , height/8 , width/8)
            nn.SiLU(),
            # (batch_size , 512 channels , height/8 , width/8) -> (batch_size , 8 channels , height/8 , width/8)
            nn.Conv2d(512,8,kernel_size=3,padding=1), ## Bottleneck of the encoder !!!! 
            # (batch_size , 8 channels , height/8 , width/8) -> (batch_size , 8 channels , height/8 , width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0),
        )
         
        
    def forward(self,x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor:
        # x     : (batch_size , channels , height , width)
        # noise : (batch_size , 4 , height/8 , width/8)
        
        ## run the images through the sequential model ==> compressed versions of the images 
        for module in self : 
            if getattr(module,'stride',None) == (2,2):
                # Padding (left,right,top,bottom) 
                x = F.pad(x,(0,1,0,1)) # => Asymmetric padding for images resulting from conv with stride=2 
            x = module(x)
        # (batch_size , 8 , height/8 , width/8) => two tensors of shape (batch_size , 4 , height , height/8 , width/8)  
        mean , log_var = torch.chunk(x,2,dim=1) # divide into two tensors at dim=1 (8 -> 4 -- 4)
        
        log_var = torch.clamp(log_var , -30 , 20)
        var = log_var.exp()
        std = var.sqrt()
        # Z = N(0,1) -> X = N(mean , std) ? 
        # X = mean + std*Z
        x = mean + std * noise
        # scaling by a constant
        x *= 0.18215
        
        return x
