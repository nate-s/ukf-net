import numpy as np
import torch
import torch.nn as nn
import numpy as np


"""
This is a utility module I copied DIRECTLY from the nvidia github page. I wrote my own before using the built in squeeze excite blocks from pytorch, but this
was easier ot prototype with in the moment.
"""

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
    
    
# FE
"""
This is a utility module I ALSO copied DIRECTLY from the nvidia github page. I wrote my own before as well but once again it was easier 
to prototype in the moment with
"""

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
    
    
    
class pred_module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_point = nn.Conv2d(dim, dim//2, 1)
        self.FE = FeatExtract(dim, False)
        self.FE2 = FeatExtract(dim//2, True)
        self.conv_red = nn.Conv2d(dim//2, 1, 3, 2, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.FE(x)
        x = self.conv_point(x)
        x = self.FE2(x)
        x = self.act(x)
        x = self.conv_red(x)
        return x
    
    
    
"""
Depth network
"""
    
class x_net_decomp(nn.Module):
    # dim, kernel, stride, dilation, latent, kernel_p, stride_p, inv_k, inv_s
    def __init__(self, dim, kernel, stride, padding, kernel_pool, stride_pool, padding_pool, kernel_inv, stride_inv, padding_inv):
        super().__init__()
        """
        This class is the autoencoder pipeline that is pre trained to predict depth maps from single images, it's latent space is given to the odometry network 
        note: This is a cumbersome network with patchwork design, but gave me the best performance on the NYUv2 dataset I could personalyl achieve. It is not intended for genuine use, and an imediate improvment would be to use a teacher student network to reduce the size and relax overfitting.

        """

        # Encoding layer 1

        self.act_01 = nn.GELU()
        self.conv_01  = nn.Conv2d(dim[0], dim[1], kernel[0], stride[0], padding[0])
        self.conv_11 =  nn.Conv2d(dim[1], dim[1], kernel[0], stride[0], padding[0])
        self.pool_01 = nn.MaxPool2d(kernel_pool[0], stride_pool[0], padding_pool[0])
        self.drop_01 = nn.Dropout(p = 0.25)
        self.fe_01 = FeatExtract(dim[1])

        # Encoding layer 2

        self.act_02 = nn.GELU()
        self.conv_02  = nn.Conv2d(dim[1], dim[2], kernel[1], stride[1], padding[1])
        self.conv_12 =  nn.Conv2d(dim[2], dim[2], kernel[1], stride[1], padding[1])
        self.pool_02 = nn.MaxPool2d(kernel_pool[1], stride_pool[1], padding_pool[1])
        self.drop_02 = nn.Dropout(p = 0.25)
        self.fe_02 = FeatExtract(dim[2])

        # Encoding layer 3

        self.act_03 = nn.GELU()
        self.conv_03  = nn.Conv2d(dim[2], dim[3], kernel[2], stride[2], padding[2])
        self.conv_13 =  nn.Conv2d(dim[3], dim[3], kernel[2], stride[2], padding[2])
        self.pool_03 = nn.MaxPool2d(kernel_pool[2], stride_pool[2], padding_pool[2])
        self.drop_03 = nn.Dropout(p = 0.25)
        self.fe_03 = FeatExtract(dim[3])

        # Encoding layer 4

        self.act_04 = nn.GELU()
        self.conv_04  = nn.Conv2d(dim[3], dim[4], kernel[3], stride[3], padding[3])
        self.conv_14 =  nn.Conv2d(dim[4], dim[4], kernel[3], stride[3], padding[3])
        self.pool_04 = nn.MaxPool2d(kernel_pool[3], stride_pool[3], padding_pool[3])
        self.drop_04 = nn.Dropout(p = 0.25)
        self.fe_04 = FeatExtract(dim[4])
        

        # Latent layer
        self.l_drop = nn.Dropout(0.25)
            # re-expand
        self.latent_01 = nn.Conv2d(dim[4], dim[3], kernel[4], stride[4], padding[4])
        self.bi_up_01 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.latent_02 = nn.Conv2d(dim[3], dim[2], kernel[4], stride[4], padding[4])
        self.bi_up_02 = nn.UpsamplingBilinear2d(scale_factor=2)
            # re-shrink
        #self.latent_03 = nn.Conv2d(dim[2], dim[3], kernel[4], stride[4], padding[4])
        self.latent_03 = nn.Conv2d(dim[2], dim[3], 3, 1, padding[4])
        self.latent_04 = nn.Conv2d(dim[3], dim[4], 3, 2, padding[4])
        
        self.l = nn.Conv2d(dim[4], dim[4], 3, 2, padding[4])
        self.ll = nn.Conv2d(dim[4], dim[4], 3, 1, padding[4])
        
        self.l_fe1 = FeatExtract(dim[4])
        self.l_fe2 = FeatExtract(dim[4], True)
        
        self.act_05 = nn.GELU()
        
        self.bn_02 = nn.BatchNorm2d(dim[2])
        self.bn_03 = nn.BatchNorm2d(dim[3])
        

        # Decoder layer 1

        self.d_drop = nn.Dropout(0.25)
        self.p_0 = pred_module(dim[4])
        self.inv_01 =  nn.ConvTranspose2d(dim[4], dim[4], kernel_inv[0], stride_inv[0], padding_inv[0])
        self.bn_i1 = nn.BatchNorm2d(dim[4])
        self.conv_i01 = nn.Conv2d(dim[4], dim[3],  kernel[5], stride[5], padding[5])
        self.conv_i02 = nn.Conv2d(dim[3], dim[3],  kernel[5], stride[5], padding[5])
        self.act_06 = nn.GELU()
        self.p_1 = pred_module(dim[3])

        # Decoder layer 2

        self.inv_02 = nn.ConvTranspose2d(dim[3], dim[3], kernel_inv[1], stride_inv[1], padding_inv[1])
        self.bn_i2 = nn.BatchNorm2d(dim[3])
        self.conv_i11 = nn.Conv2d(dim[3], dim[2],  kernel[5], stride[5], padding[5])
        self.conv_i12 = nn.Conv2d(dim[2], dim[2],  kernel[5], stride[5], padding[5])
        self.act_06 = nn.GELU()
        self.p_2 = pred_module(dim[1])
        # Decoder layer 3

        self.inv_03 = nn.ConvTranspose2d(dim[2], dim[1], kernel_inv[2], stride_inv[2], padding_inv[2])
        self.bn_i3 = nn.BatchNorm2d(dim[1])
        self.conv_i21 = nn.Conv2d(dim[1], dim[1],  kernel[5], stride[5], padding[5])
        self.conv_i22 = nn.Conv2d(dim[1], dim[0],  kernel[5], stride[5], padding[5])
        self.act_06 = nn.GELU()
        self.act_00 = nn.Sigmoid()
        
        self.var_ = nn.Linear(1200*2,1200*2)
        self.mu_ = nn.Linear(1200*2,1200*2)


    def encode(self, image, drop):
        
        L01 = self.act_01( self.conv_11( self.act_01( self.conv_01(image) ) ) )
        L11 = self.fe_01( L01 ) #self.pool_01( L01 )
        if drop:
            L11 = self.drop_01( L11 )  

        # Layer 2: 224 -> 112
        L02 = self.act_02( self.conv_12( self.act_02( self.conv_02(L11) ) ) )
        L12 = self.fe_02( L02 ) #self.pool_02( L02 ) 
        if drop:
            L12 = self.drop_02( L12)

        # Layer 3: 112 -> 56
        L03 = self.act_03( self.conv_13( self.act_03( self.conv_03(L12) ) ) )
        L13 = self.fe_03( L03 ) #self.pool_03( L03 )
        if drop:
            L13 = self.drop_03( L13 )

        # Layer 4: 56 -> 28
        L04 = self.act_04( self.conv_14( self.act_04( self.conv_04(L13) ) ) )
        L14 = self.fe_04( L04 ) #self.pool_04( L04 )
        if drop:
            L14 = self.drop_04( L14 ) 

        return L14, L13, L12, L11


                         
    def latent(self, latent_in,  o3, o2, drop, latent_noise):

        if drop:
            latent_in = self.l_drop(latent_in)
        L01 = self.act_05( self.latent_01( latent_in ) ) # might not makes sense to GELU before a bilinear upsample
    
        L01 = self.bi_up_01( L01 )
        
        if drop:
            L01 = self.l_drop(L01)
        L02 = self.act_05( self.latent_02( L01 + o3 ) ) 
        L02 = self.bn_02(L02) # Batch normalization after a skip connection has been shown empircally to reduce error and improve convergence (for resnet models)
        L02 = self.bi_up_02( L02 )
        
        L03 = self.act_05( self.latent_03( L02 + o2 ) )
        L03 = self.bn_03(L03)
        L04 = self.latent_04( L03 )

        if drop:
            L04 = self.l_drop(L04)
    
        if 0:
            o1 = self.l( L04 )
            o2 = self.ll( o1 )
        else:
            o1_ = self.l_fe1( L04 )
            if 0:
                batch_size = o1_.shape[0]
                o1_ = o1_.reshape(batch_size, 256,-1)
                o1_ = self.var_(o1_)*latent_noise + self.mu_(o1_)
                o1_ = o1_.reshape(batch_size,256*2,30,40)
            o2 = self.l_fe2( o1_ )
            
            #print(o1.shape, o2.shape, o1_.shape, o2_.shape)
        
        #print('Latent shapes first to last:', latent_in.shape, L01.shape, L02.shape, L03.shape, L04.shape, o1_.shape, o2.shape)#256x30x40

        return o2, L04, L03, 
    
    
    def decode(self, l, l02, l03, l04, drop):
        # Note: probabl need to reshape the skip connections? 
        pred = False
        # Decode layer 1: 
        t = self.inv_01( l )
        L_01 = t + l02
        L_01 = self.bn_i1(L_01)
        L_02 = self.act_06( self.conv_i01( L_01 ) ) 
        L_03 = self.act_06( self.conv_i02( L_02 ) )


        # Decode layer 2: 
        if drop:
            L_03 = self.d_drop(L_03)
        t2 = self.inv_02( L_03 )
        L_11 = t2 + l03
        L_11 = self.bn_i2(L_11)

        L_12 = self.act_06( self.conv_i11( L_11 ) ) 
        L_13 = self.act_06( self.conv_i12( L_12 ) )


        # Decode layer 3:
        if drop:
            L_13 = self.d_drop(L_13)
        t3 = self.inv_03( L_13 )
        L_21 = t3 + l04
        L_21 = self.bn_i3(L_21)


        L_22 = self.act_06( self.conv_i21( L_21 ) ) 
        L_23 = ( self.conv_i22( L_22 ) )

        # Then conv until you get original image + depth layer filter
        if pred:
            Y_1 = self.p_1(L_11)
            Y_2 = self.p_2(L_21)
            Y_0 = self.p_0(L_01)

        return L_23 #, Y_2, Y_1, Y_0
    

    def forward(self, x, drop, noise):

        o4, o3, o2, o1 = self.encode( x, drop ) # deep to shallow latent
        L, _o3, _o2 = self.latent( o4, o3, o2, False, noise)
        x = (self.decode( L, _o3, _o2, o1, drop ))

        return x, _o3, _o2
    
    
    
    


class vio_net(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        """
        Odometry network, takes in the previous 3 latent spaecs of the depth network and predicts the change in position between frames. As of right now that is the only thing it predicts. I found that predicting velocity and accelertion hampered performance while being easily extarctable from the delta X, so those values are left out for now. I don't think it would be wise to add a new UKF for each of these translational predicitons, howveer, if you were to add orientation I think it would be smart to use a new UKF and dynamic model.
        """
        
        b_x = torch.zeros(6)
        self.biases_x = nn.Parameter(data=b_x,requires_grad=True)
        b_dx = torch.zeros(3)
        self.biases_dx = nn.Parameter(data=b_dx,requires_grad=True)
        
        self.c1 = nn.Conv2d(512*batch_size, 512, 3, 2, 1)
        self.pc1 = nn.Conv2d(512, 512, 1)
        self.c2 = nn.Conv2d(512, 256, 3, 2, 1)
        self.pc2 = nn.Conv2d(256, 256, 1)
        
        self.FE1 = FeatExtract(512, True)
        self.FE2 = FeatExtract(256, True)
        
        
        self.act1 = nn.GELU()
        if 0:
            self.crot = nn.Conv2d(256, 128, 3, 2, 1)
            self.linrot1 = nn.Linear(1024*2, 512)
            self.linrot2 = nn.Linear(512, 512//2)
            self.linrot3 = nn.Linear(512//2, 3*(batch_size-1)) # (batch_size-1)*
            self.linang2 = nn.Linear(512, 512//2)
            self.linang3 = nn.Linear(512//2, 3*(batch_size-2)) # (batch_size-1)*

        self.cpose = nn.Conv2d(256, 128, 3, 2, 1)
        self.linpose1 = nn.Linear(1024*2, 512)
        self.linpose2 = nn.Linear(512, 512//2)
        self.linpose3 = nn.Linear(512//2, 3*(batch_size-1)) # (batch_size-1)*
        self.linvel2 = nn.Linear(512, 512//2)
        self.linvel3 = nn.Linear(512//2, 3*(batch_size-2)) # (batch_size-1)*
                              
            
    def encode(self, x):
        
        # Encode
        x = self.c1(x)
        x = self.FE1(self.pc1(x).unsqueeze(0)).squeeze()
        x = self.act1(self.c2(x))
        x = self.FE2(self.pc2(x).unsqueeze(0)).squeeze()
        x = self.act1(x)
        
        # Encode to angular and translational
        #x_r = self.crot(x)
        #x_r = x_r.flatten()
        x_r=0
        x_p = self.cpose(x)
        x_p = x_p.flatten()
        
        # Make it smaller 
        #x_r = self.act1(self.linrot1(x_r))
        x_p = self.act1(self.linpose1(x_p))
        
        x_a = x_r
        #x_r = self.linrot2(x_r)
        #x_a = self.linang2(x_a)
        
        x_v = x_p
        x_p = (self.linpose2(x_p))
        x_v = (self.linvel2(x_v))
        
        return x_r, x_p, x_a, x_v # lets say dx, do, ddx, ddo
        
    def forward(self, x_r, x_p, x_a, x_v):
        # This is literally just a signle linear layer that outputs a 2x3 matrix corresponding to the change in positionbetween frames K,K+1, and K+2
        x_ = self.linpose3(x_p) #+ self.biases_x (for some reason the predicitons are consistently biased by -0.15, so I thought this might help. It did not...
        # x_vdot is ignored 
        x_vdot = self.linvel3(x_v)
        
        
        return x_.reshape(2,3), x_vdot.reshape(1,3), 0, 0# x_rdot.reshape(2,3), x_adot.reshape(1,3)
        