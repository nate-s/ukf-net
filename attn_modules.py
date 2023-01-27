import torch
import torch.nn as nn
import numpy as np

class SE(nn.Module):
    """
    Squeeze and excitation block
    """
    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x

class AttnLocal(nn.Module):
    def __init__(self, 
                 dim, 
                 heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        
        super().__init__()
        
        """
        This class performs local attention on one dimensional vectors
        """
        self.dim = dim
        self.heads = heads
        self.P = window_size
        self.qkv_bias = qkv_bias
        self.bias = 0

        self.kqv = nn.Linear(dim, dim*3)
        self.sf = nn.Softmax(dim = -1)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, _):
        
        # x = [Batch size (layer dimension), Num windows = (window size^2 when image W/H is divisible by integers), Embeded layer dim (128)] (for example, first latentlayer shape = 64, 196 (14*14), 128)
        B_, N, C = x.shape
        kqv = self.kqv(x).reshape(x.shape[0],x.shape[1],3,self.heads, C // self.heads).permute(2,0,3,1,4)
        
        k = kqv[:][0].squeeze()
        q = kqv[:][1].squeeze()
        v = kqv[:][2].squeeze()
        q = (q / np.sqrt(self.dim)) # Dim is 256

        if self.qkv_bias == True:
            attention = self.sf((k @ q.transpose(-2,-1)) + self.bias)
        else:
            attention = self.sf((k @ q.transpose(-2,-1)))
        attention = self.attn_drop(attention)
        
        out_1 = (attention @ v).reshape(B_, N, C)
        out_2 = self.proj(out_1)
        x = self.proj_drop(out_2)
       
        return x.squeeze() # Returns: [B_,N,C] tensor

class AttnGlobal(nn.Module):
    def __init__(self, 
                 dim, 
                 heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        
        super().__init__()
        
        """
        This class performs global attention on one dimensional vectors. Literraly the same as above, just global Q now
        """
        self.it = 0
        self.dim = dim
        self.heads = heads
        self.P = window_size
        
        self.kv = nn.Linear(dim, dim*2)
        self.sf = nn.Softmax(dim = -1)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, q_global):
        
        # x = [N, window_size, window_size]
        B_, n, p = x.shape
        
        kqv = self.kv(x).reshape(x.shape[0],x.shape[1],2,-1).permute(2,0,1,3)
        
        k = kqv[:][0].squeeze()
        v = kqv[:][1].squeeze()
        # This is the global layer
    
        q_global = (q_global / np.sqrt(self.dim)) # Dim is 256
        
        attention = self.sf((k @ q_global.transpose(-2,-1)))
        attention = self.attn_drop(attention)
        # This is global layer
        #assert attention @ v, print("Attn: ", attention.shape, "V: ", v.shape)
        #assert attention.shape[3] == v.shape[1], print("OH SHIT: ", k.shape, q_global.shape)
        out_1 = attention @ v
        out_2 = self.proj(out_1)
        x = self.proj_drop(out_2)

        return x # Returns: [1,B_,N,Dim] tensor, idk why the 1 is there or where it came from...


class AttnDyn(nn.Module):
    def __init__(self, 
                 dim, 
                 heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        
        super().__init__()
        
        """
        This class performs global attention on one dimensional vectors. Literraly the same as above, just global Q now
        """
        self.it = 0
        self.dim = dim
        self.heads = heads
        self.P = window_size
        
        self.q = nn.Linear(dim, dim)
        self.sf = nn.Softmax(dim = -1)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, kv):
        
        # x = [N, window_size, window_size]
        B_, n, p = x.shape

        q = self.q(x).squeeze() #.reshape(x.shape[0],x.shape[1],2,-1).permute(2,0,1,3)
        k = kv[0,0:8,:].squeeze().permute(1,0)
        v = kv[0,8:,:].squeeze()
        # This is the global layer
    
        q_global = (q / np.sqrt(self.dim)) # Dim is 256
        
        attention = self.sf((k @ q_global))
        attention = self.attn_drop(attention)
        # This is global layer
        #assert attention @ v, print("Attn: ", attention.shape, "V: ", v.shape)
        #assert attention.shape[3] == v.shape[1], print("OH SHIT: ", k.shape, q_global.shape)
        out_1 = torch.matmul(attention, v.permute(1,0)).permute(1,0)
        out_2 = self.proj(out_1).unsqueeze(0)
        x = self.proj_drop(out_2)

        return x # Returns: [1,B_,N,Dim] tensor, idk why the 1 is there or where it came from...

class AttnBlock(nn.Module):
    def __init__(self,
                 dim, 
                 heads,
                 attn_type,
                 window_size = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        """
        This class wraps an attention module with a skip connection and feed forward MLP block
        """
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        
        self.attn = attn_type(dim = dim, heads = heads, window_size = window_size)
        
        self.lin_map = nn.Linear(dim, dim)
        
    def forward(self, x, q_global):
        """
        Normalize
        Partition
        Attend
        Shortcut
        MLP
        Normalize
        """
        skip = x
        x = self.norm_1(x)
        attn =self.attn.forward(x, q_global) #Input: partioned window [NxD] tensor, Returns: [NxD] tensor

        if attn.shape[1] > 256:
            attn_mean = torch.mean(attn, dim=1)
            latent = attn_mean + skip
        else:
            latent = attn + skip

        out = latent + self.lin_map(self.norm_2(latent))
        return out

class attention_local_global_layer(nn.Module):
    def __init__(self, dim, depth, heads, AttnType):
        super().__init__()
        """
        Fuck it this is just gonna hard sandwich a local MHSA with a GMHA
        
        This is the backbone of my pattented Odometry Guided generative Relocalization (something with E) or OGRE for short. Get it? Cause it's SO LAYERED... ha
        This ogres block can be repeated and stacked as many times as desired, but each attentonlayer creates (P^2)*D + N*D parameters so don't go hog wild
        """
        heads = heads
        depth = depth
        self.reduce = False
        self.d = dim
        """
        self.ogres = nn.ModuleList([AttnBlock(attn_type = AttnLocal if (i % 2 == 0) else AttnGlobal, 
                                              dim = dim, 
                                              heads = heads)
                                    for i in range(depth)]) #Note: this is a stack of local->global transformer layers. It can go as deep as you want but keeps the same Dim per class isntance
        """
        self.ogres = nn.ModuleList([AttnBlock(attn_type = AttnType,
                                              dim = dim, 
                                              heads = heads)
                                    for i in range(depth)]) #Note: this is purely local self attention 

        self.feat_extract = FeatExtract(dim = dim, keep_dim = True)
        self.reduce = False
        #self.gloabl_q = global_q_gen(stuff)
    def forward(self, x, q_global):
        
        #q_global = self.global_q(x)
        for layer in self.ogres:
            x = layer(x, q_global)
        
        if self.reduce:
            a, b, c = x.shape
            x = x.permute(0,2,1)
            x = self.feat_extract( x.reshape( a, c, int(np.sqrt(b)), int(np.sqrt(b)) ) ).reshape( a, b, -1 ).permute( 1,0,2 )
            
        # Note: at the end of every pass we reduce size of the input except for the last layer of the transformer. This is done by the [conv(3)->GELU()->SE()->CONV(1)] -> conv_down((dim, dim*2)
        return x


class encoder_attn(nn.Module):
    def __init__(self, numVstates):
        """
        This will be the single class called in each training pass. Has to handle averything for a single transformer. 
        We have three transformers each taking a different slice of autoencoder latent space
        
        input:
            depth: the depth of each transformer layer all with dimension D
            head: the number of heads in each local+global pair
        """
        super().__init__()

        dim_encoder = 8 # Dimension at AttnBlock 
        d_encoder = 2
        num_heads_encoder = 1
        layer_depth_encoder = 2
        self.max_sequence_length = 15 # how many time steps we will use the same key before restarting

        dim_dynamics = 256
        d_dynamics = 2
        num_heads_dynamics = 1
        layer_depth_dynamics = 2
        self.key = 0
        emb = torch.rand((23, 4, 8))
        self.e = nn.Parameter(data = emb, requires_grad=True)
        emb_q = torch.rand((23, 1, 8))
        self.e_q = nn.Parameter(data = emb_q, requires_grad=True)

        at_l = AttnLocal
        at_g = AttnGlobal
        at_dyn = AttnDyn

        self.q_map = nn.Linear(32 + self.emb_dim + numVstates,256)
        self.dynamic_map = nn.Linear(8,1)


        self.encode = nn.ModuleList([attention_local_global_layer(dim = dim_encoder, depth = d_encoder, heads = num_heads_encoder, AttnType = at_l)
                                for i in range(layer_depth_encoder)
                                ])

        self.dynamics = nn.ModuleList([attention_local_global_layer(dim = dim_dynamics, depth = d_dynamics, heads = num_heads_dynamics, AttnType = at_dyn)
                                for i in range(layer_depth_dynamics)
                                ])

    def key_emb(self,x):
        """
        The forward takes either an image or a flat vector of arbitrary size, patch embeds it
        then runs it through the corresponding attention network, the norms+avgpools+flattens it again
        """
        x = x.permute(2,0,1)
        x = torch.concat((x, self.e), dim = 0).permute(1,0,2)
        for layer in self.encode:
            x = layer(x, x)
        
        x = x.reshape(8,64,-1).flatten(-2,-1)

        return x
       
    def dynamic_update(self, q, k, v):

        q = torch.concat((q.permute(2,0,1), self.e_q), dim = 0).permute(2,1,0) #.flatten()
        q = self.q_map(q).squeeze().unsqueeze(0)
        kv = torch.concat((k.unsqueeze(0),v),dim=1)

        for layer in self.dynamics:
            q = layer(q,kv)
        
        x = self.dynamic_map(q.permute(0,2,1))
            
        return x