import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math 


class SelfAttention(nn.Module):
    def __init__(self,n_heads:int,d_embed:int , in_bias=True,out_bias=True):
        super().__init__()
        self.in_proj  = nn.Linear(d_embed,3*d_embed,bias=in_bias) ## the three matrices w_q w_k w_v as one big matrix 
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self,x:torch.Tensor,causal_mask=False): 
        # x (batch_size,seq_len,dim)
        in_shape = x.shape
        batch_size , seq_len , d_embed = in_shape # seq_len = height * width
        # d_embed = n_heads * d_heads
        intermediate_shape = (batch_size,seq_len,self.n_heads,self.d_heads)
        # (batch_size,seq_len,dim) -> (batch_size , seq_len , dim*3) -> 3 tensors of shape (batch_size , seq_len , dim)
        q,k,v = self.in_proj(x).chunk(3,dim=-1)
        ## each head will see the whole sequence but with a limited part of the embedding
        q = q.view(intermediate_shape).transpose(1,2) # (batch_size , seq_len , n_heads , d_heads) -> (batch_size , n_heads , seq_len , d_heads)
        k = k.view(intermediate_shape).transpose(1,2) # (batch_size , seq_len , n_heads , d_heads) -> (batch_size , n_heads , seq_len , d_heads)
        v = v.view(intermediate_shape).transpose(1,2) # (batch_size , seq_len , n_heads , d_heads) -> (batch_size , n_heads , seq_len , d_heads)
        
        #@@@@ Calculating the attention @@@@#
        
        weight = q @ k.transpose(-1,-2) # (batch_size , n_heads , seq_len , seq_len)
        if causal_mask : 
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1) ## Upper triangle
            weight.masked_fill_(mask,-torch.inf)
        
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight , dim=-1)
        # (batch_size , n_heads , seq_len , seq_len) @ (batch_size , n_heads , seq_len , d_heads) -> (batch_size , n_heads , seq_len , d_heads)
        output = weight @ v
        output = output.transpose(1,2) # (batch_size , seq_len ,n_heads , d_heads)
        output = output.reshape(in_shape)
        
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module) :
    
    def __init__(self,n_heads:int,d_embed:int,d_cross:int,in_bias=True,out_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed,d_embed,bias=in_bias)
        self.k_proj = nn.Linear(d_cross,d_embed,bias=in_bias)
        self.v_proj = nn.Linear(d_cross,d_embed,bias=in_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads
    
    def forward(self,x,context) : 
        # x : latent (batch_size , seq_len_Q , dim)
        # context (batch_size , seq_len_KV , dim) = (batch_size , 77 , 768) 

        input_shape = x.shape
        batch_size , seq_len , dim_embed = input_shape
        inter_shape = (batch_size,-1,self.n_heads,self.d_heads)

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        q = q.view(inter_shape).transpose(1,2)
        k = k.view(inter_shape).transpose(1,2)
        v = v.view(inter_shape).transpose(1,2)
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight,dim=-1)
        
        output = weight @ v 
        
        output = output.transpose(1,2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        
        return output