import torch
import torch.nn as nn 
from torch.nn import functional as F 
from attention import SelfAttention , CrossAttention


class UNET_residual_block(nn.Module) : 
    def __init__(self ,in_channels:int , out_channels:int , n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels , out_channels , kernel_size=3,padding=1) 
        
        self.linear_time = nn.Linear(n_time,out_channels) # time embedding
        
        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels , out_channels , kernel_size=3 , padding=1)

        if in_channels == out_channels :
            self.residual_layer = nn.Identity()
        else : 
            self.residual_layer = nn.Conv2d(in_channels , out_channels,kernel_size=1,padding=0)
    def forward(self,feature,time) : 
        # feature (batch_size , in_channels , height , width)
        # time (1 , 1280)
        
        resid = feature 
        feature = self.groupnorm_feature(feature) 
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        # (batch_size , out_channels,height,width)
        return merged + self.residual_layer(resid)
                       
class UNET_AttentionBlock(nn.Module):
    def __init__(self , n_heads:int , n_embed:int , d_context=768):
        super().__init__()
        channels = n_heads * n_embed
        self.groupnorm   = nn.GroupNorm(32,channels,eps=1e-6) 
        self.conv_input  = nn.Conv2d(channels,channels,kernel_size=1,padding=0) 
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads , channels , in_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads , channels ,d_context, in_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels , 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels , channels)
        self.conv_output    = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
    def forward(self,x,context):
        # Latent : x (batch_size , features , height , width)        
        # Prompt : context (batch_size , seq_len , dim=768)
        resid_1 = x 
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n,c,h,w = x.shape
        # (batch_size , features , height , width) -> (batch_size , features , height * width)
        x = x.view((n,c,h*w))
        # (batch_size , features , height , width) -> (batch_size ,  height * width , features)
        x = x.transpose(-1,-2)
        ## Normalization + self attention
        resid_2 = x 
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += resid_2
        resid_2 = x 
        
        # Normalization + cross attention
        x = self.layernorm_2(x)
        x = self.attention_2(x,context) ## (cross attention between the prompt and the image)
        x += resid_2
        
        resid_2 = x 
        x = self.layernorm_3(x)
        x , gate = self.linear_geglu_1(x).chunk(2,dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += resid_2 
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))

        return self.conv_output(x) + resid_1

class Upsample(nn.Module):
    def __init__(self , channels:int) : 
        super().__init__()
        self.conv = nn.Conv2d(channels , channels , kernel_size=3,padding=1)

    def forward(self , x):
        # (batch_size , features , height , width) ->  (batch_size , features , height*2 , width*2)
        x = F.interpolate(x , scale_factor=2,mode='nearest')
        return self.conv(x)
            
class TimeEmbedding(nn.Module):
    def __init__(self , n_embed:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed,4*n_embed)
        self.linear_2= nn.Linear(4*n_embed,4*n_embed)

    def forward(self , x:torch.Tensor) -> torch.Tensor : 
        # x (1 , 320)
        x = self.linear_1(x)
        x = F.silu(x) 
        x = self.linear_2(x) 
        # (1 , 1280)
        return x 

class SwitchSequential(nn.Sequential):
    def forward(self, x , context, time): 
        for layer in self : 
            if isinstance(layer,UNET_AttentionBlock) : 
                x = layer(x,context)
            elif  isinstance(layer , UNET_residual_block) : 
                x = layer(x,time)
            else : 
                x = layer(x)
        return x 
     
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (batch_size , 4 , height/8 , width/8) -> (batch_size , 320 , height/8 , width/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(UNET_residual_block(320,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_residual_block(320,320),UNET_AttentionBlock(8,40)),
            # (batch_size , 320 , height/8 , width/8) -> (batch_size , 320 , height/16 , width/16) 
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residual_block(320,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residual_block(640,640),UNET_AttentionBlock(8,80)),
            # (batch_size , 320 , height/16 , width/16) -> (batch_size , 320 , height/32 , width/32)
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residual_block(640,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residual_block(1280,1280),UNET_AttentionBlock(8,160)),
            # (batch_size , 1280 , height/32 , width/32) -> (batch_size , 1280 , height/64 , width/64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residual_block(1280,1280)),
            SwitchSequential(UNET_residual_block(1280,1280)),# (batch_size , 1280 , height/64 , width/64) 
        ]) 
        #@@ BottleNeck of the UNET 
        self.bottleneck = SwitchSequential(
            UNET_residual_block(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_residual_block(1280,1280),
        )
        self.decoders = nn.ModuleList([
            # (batch_size , 2560 , height/64 , width/64) -> (batch_size , 1280 , height/64 , width/64) 
            # // 2560= 1280 + 1280 because of the residual connection  
            SwitchSequential(UNET_residual_block(2560,1280)),
            SwitchSequential(UNET_residual_block(2560,1280)),
            SwitchSequential(UNET_residual_block(2560,1280) , Upsample(1280)), # (batch_size , 1280 , height/32 , width/32) 
            SwitchSequential(UNET_residual_block(2560,1280) , UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residual_block(2560,1280) , UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residual_block(1920,1280) , UNET_AttentionBlock(8,160) , Upsample(1280)), # (batch_size , 1280 , height/16 , width/16)
            SwitchSequential(UNET_residual_block(1920,640) , UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residual_block(1280,640) , UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residual_block(960,640) , UNET_AttentionBlock(8,80) , Upsample(640)), # (batch_size , 640 , height/8 , width/8)
            SwitchSequential(UNET_residual_block(960,320) , UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_residual_block(640,320) , UNET_AttentionBlock(8,40)),
            # (batch_size , 320 , height/8 , width/8) 
            SwitchSequential(UNET_residual_block(640,320) , UNET_AttentionBlock(8,40)), ## the output of the UNET that will be fed to the decoder and the scheduler     
        ])
        
    def forward(self,x,context,time) : 
        # x : (batch_size , 4 , height/8 , width/8)
        # context : (batch_size , seq_len , dim)
        # time : (1 , 1280)
        
        skip_connections = []
        for layers in self.encoders : 
            x = layers(x,context,time)
            skip_connections.append(x)
            
        x = self.bottleneck(x,context,time)

        for layers in self.decoders : 
            x = torch.cat((x,skip_connections.pop()),dim=1)
            x = layers(x,context,time)
        
        return x 
        
class unet_out_layer(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels , out_channels , kernel_size=3,padding=1)
    def forward(self,x) : 
        ## (batch_size , 320 , height/8 , width/8) -> (batch_size , 4 , height/8 , width/8)
        x = self.groupnorm(x)
        x= F.silu(x)
        x = self.conv(x) # (batch_size , 4 , height/8 , width/8)
        
        return x 
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = unet_out_layer(320,4)
        
    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor) :
        # latent (batch_size , 4 , height/8 , width/8)
        # context (batch_size , seq_len , dim )
        # time (1,320)
        # (1,320) --> (1 , 1280)
        time= self.time_embedding(time) 
        
        # latent (batch_size , 4 , height/8 , width/8) -> latent (batch_size , 320 , height/8 , width/8)
        output = self.unet(latent,context,time)
        
        # latent (batch_size , 320 , height/8 , width/8) -> latent (batch_size , 4 , height/8 , width/8)
        ## the UNET will predict how much noise in the image , the scheduler will remove the predicted noise and than put back the image to the UNET 
        ## so the output of the UNET should have the same shape as the input
        output = self.final(output)

        return output        