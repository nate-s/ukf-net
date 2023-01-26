import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import attn_modules as attn
import scipy.linalg

# sqrtm
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
          from https://github.com/steveli/pytorch-sqrtm
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

# Returns: [2NxN] sigma points and P_k
def sigma_points(x, P_u, Q_k):
    """
    Inputs: [1xN] vector, the previous covaroiance matrix P_u[+|k-1], and process noise Q_k
    Returns: [2NxN] vector of sigma points, and a [NxN] matrix of covariance P_u[-|k]
    """

    sigma = x.repeat(1,1,1,x.shape[2])
    print('Sigma points for',x.shape[2])

    p = sqrtm(x.shape[2]*torch.abs(P_u.squeeze()))    
    sigma_plus = sigma + p
    sigma_minus = sigma - p

    sigma = torch.concat((sigma_plus, sigma_minus), dim=3)
    scale_factor = 1/(2*x.shape[2])
    x_bar = torch.mean(sigma, dim=3)
    sum = torch.zeros((x.shape[1],x.shape[2],x.shape[2]))
    extra = np.zeros((x.shape[2]*2,x.shape[2],x.shape[2]))

    for i2 in range(sigma.shape[1]):
        for i1 in range(sigma.shape[3]):
            vecp = sigma[0,0,:,i1].unsqueeze(1)
            c = (vecp.squeeze()-x_bar[0,i2].squeeze().T)
            sum[i2] += torch.matmul(c.unsqueeze(-1), c.unsqueeze(0))

    P_k = sum*scale_factor + (Q_k**2)*0.01
    
    return P_k, sigma.permute(0,1,3,2) #, extra # The sigma points for X_hat[k|k-1] and the cross variance matrix for X_hat[k|k-1]
    #return sigma, x_bar

# Returns: Updated X post, P_k post, K_k for record keeping
def UKF_update(x_sigma, y_sigma, R_k, P_u, Y):

    """
    P_xx, p_hat, sigma_Y_p, state_Y, self.R_kp
    Inputs: uhhhhhhh... the [2NxN] sigma points for X, the [2NxN] outputs Y from the equation H(X_sigma_i) = Y_sigma_i,
    the measurment [1xN] noise R_k, the previous [NxN] covariance matrix P_u, and the actual [1xN] state estimate Y

    Returns: updated vector X_posteriori, the updated covariance matrix P_k+, and Kalman gain K_k
    p = sqrtm(x.shape[2]*torch.abs(P_u.squeeze()))    
    sigma_plus = sigma + p
    sigma_minus = sigma - p
    """
    print('UKF for', x_sigma.shape[2])
    
    x_bar = torch.mean(x_sigma, dim=2) # This give a Nx1 vector of the means
    y_bar = torch.mean(y_sigma, dim=2)
    scale_factor = 1/(x_sigma.shape[2])

    P_xy = torch.zeros((x_sigma.shape[1],x_bar.shape[2],x_bar.shape[2]))
    P_yy = P_xy
    
    for i2 in range(P_xy.shape[0]):
        for i1 in range(x_sigma.shape[3]):
            vec_x = x_sigma[0,0,i1,:]#.unsqueeze(1)
            vec_y = y_sigma[0,0,i1,:]#.unsqueeze(1)
            
            c_1 = (vec_x.squeeze()-x_bar[0,i2].squeeze().T)
            c_2 = (vec_y.squeeze()-y_bar[0,i2].squeeze().T)

            P_xy[i2] += torch.matmul(c_1.unsqueeze(-1), c_2.unsqueeze(0)) * scale_factor
            P_yy[i2] += torch.matmul(c_2.unsqueeze(-1), c_2.unsqueeze(0)) * scale_factor # R_k is sigma so convert to sigma^2

    P_yy = P_yy + (R_k**2)

    K_k = torch.matmul(P_xy, torch.inverse(P_yy))
    if (K_k.shape[0] > 1):
        K_k = torch.mean(K_k, dim=0)
   
    res = (Y - y_bar).squeeze().T

    X_k_posterior = x_bar.squeeze() + torch.matmul(K_k, res).unsqueeze(0).permute(0,2,1).squeeze()
   
    P_k_posterior = P_u - torch.matmul(K_k.squeeze(), torch.matmul(P_yy.squeeze(), K_k.squeeze().T)).unsqueeze(0)

    return X_k_posterior, P_k_posterior, K_k # Returns X_k|k, P_k|k, K_k

# Returns: X_hat
class dynamic_model(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Module predicts the current states based on learned and physical dynamic models. Is run between frames
        Inputs: Vehicle state time k-1, latent features time k-1, dt
        Returns: Vehicle state prediciton X_k_hat and latent feature prediciton X_thetaHat_k at time k
        """
        self.model = encoder_attn()# this will be the full transformer
        self.dynamics_v = dynamic_model_vehicle()
        self.sigma_ = sigma_points
        
       
    def forward(self, state_vehicle, state_latent, key, dt, P_x, P_theta,  Q_l, Q_x):
        #print('Vehicle', state_vehicle.shape, 'latent', state_latent.shape)
        P_x, vehicle_sigma = self.sigma_(state_vehicle, P_x, Q_x) # Convert vehicle state into sigma points
        x_p_hat_sigma = self.dynamics_v(vehicle_sigma, dt) # Model of vehicle dynamcis in real world with state space matrix. Input: [X_k-1_sigma, dt] Return: posteriori estimate X_k_hat
        x_p_hat = torch.mean(x_p_hat_sigma, dim = 2) # Mean of sigma points represent "true" posterior state
        
        for i1 in range(state_latent.shape[0]):
            s = state_latent[i1].unsqueeze(0)
            P_theta_, latent_sigma = self.sigma_(s, P_theta, Q_l) # Convert latent space into sigma points and pass as "Values" through transformer

        v_state_current = x_p_hat.repeat(1,8,1)
        Q = torch.concat((state_latent.reshape((1,8,32)), v_state_current), dim = 2) # [1, 8, 41]: 8 windows from 1024, size 32 per window, + 9 states
        K = key
        V = latent_sigma.squeeze().unsqueeze(0)
        x_theta = self.model.dynamic_update(Q, K, V).unsqueeze(0) #inputs: q, k, v #latent_sigma, , dt, P_theta, P_x

        for i1 in range(state_latent.shape[0]):
            s = x_theta[i1].unsqueeze(0)
            P_theta_, x_theta_sigma = self.sigma_(s, P_theta, Q_l) # Convert latent space into sigma points and pass as "Values" through transformer
        return x_theta_sigma, x_p_hat_sigma, P_x, P_theta_ # X_theta is the predicted theta state , X_phat is the predicted vehicle state Dr. arguelles, Dr. Day, Dr. Huanyu, 

    def embed(self, x_theta, x_vehicle):
        # Inputs: latent space and vehicle state for last 4 time steps [k-1,k-4]
        v_state = x_vehicle.repeat(1,8,1)
        inp = torch.concat((x_theta.squeeze(), v_state), dim = 2)

        k = self.model.key_emb(inp)
        return k


class UKF(nn.Module):
    def __init__(self, l_size, s_size, l_hist, v_hist):
        super().__init__()

        self.dynamics = dynamic_model()
        self.sigma_ = sigma_points
        self.ukf = UKF_update

        self.s_l = l_size # 1024 // (8*4)
        self.s_v = torch.eye(s_size)

        R_tensor_p = torch.eye(s_size) * torch.abs(torch.randn(s_size)) #self.s_v.shape) #torch.tensor(self.s_v, dtype=torch.double).random_(0,1) # Initialize measurement noise as random tensor [0,1]
        R_tensor_l = torch.eye(l_size) * torch.abs(torch.randn(l_size)) # torch.tensor(torch.eye(49), dtype=torch.double).random_(0,1) # Initialize measurement noise as random tensor [0,1]
        Q_tensor = torch.eye(s_size) * torch.abs(torch.randn(s_size)) # torch.tensor(6, dtype=torch.double).random_(0,1) # Initialize vehicle process noise as random tensor [0,1] for the 6dof
        Q_tensor_l = torch.eye(l_size) * torch.abs(torch.randn(self.s_l)) # torch.tensor(self.s_l, dtype=torch.double).random_(0,1) # Initialize latent process noise as random tensor [0,1]

        self.Q_k_x = nn.Parameter(data = Q_tensor, requires_grad = True) # Learned process noise parameters
        self.Q_k_l = nn.Parameter(data =Q_tensor_l, requires_grad = True) # Learned process noise parameters
        self.R_kp = nn.Parameter(data = R_tensor_p, requires_grad = True) # Learned measurement noise parameters
        self.R_kl = nn.Parameter(data = R_tensor_l, requires_grad = True) # Learned measurement noise parameters

        self.tf_len = 15
        self.embed = True
        self.key = 0

        self.energy = 0
        self.threshold = 10
        self.latent_history = l_hist
        self.vehicle_history = v_hist

    def part_1(self, image_t2, latent, x_vehicle, dt, P_xx_prior, P_xxl_prior): # Inputs: image at time t+1, vehicle state at time t, altent time t/k

         # either embed the new latents and states every tf_len steps/ based on some energy metric above a threshold
        if 1: #self.energy > self.threshold: # All good jsut need to update state history each loop
            self.key = self.dynamics.embed(self.latent_history, self.vehicle_history)

        # Priori update---------------
        p_hat, latent_hat, P_xx, P_xxl = self.F(latent, x_vehicle, P_xx_prior, P_xxl_prior, dt) # F(X:t-1|t-1) -> return state for t|t-1
        p_hat_mean = torch.mean(p_hat, dim=2)
        l_hat_mean = torch.mean(latent_hat, dim=2)

        # Measurement Y prediction----------
        sigma_Y_p, sigma_Y_l = self.H(p_hat_mean, l_hat_mean, P_xx, P_xxl) # Collect measurement prediction by passing the latent sigma points through odom net H(X)
        # Collect measurments Y-----------

        return p_hat, sigma_Y_p, P_xx, latent_hat, sigma_Y_l, P_xxl, l_hat_mean, p_hat_mean

    def part_2(self, p_hat, sigma_Y_p, P_xx, state_Y, latent_hat, sigma_Y_l, P_xxl, latent_Y):
        # Perform UKF update with X, Yhat, and Y
        # x_sigma, y_sigma, R_k, P_u, Y
        X_kpose_post, P_kp_post, K_kp = self.ukf(p_hat, sigma_Y_p, self.R_kp, P_xx, state_Y) # X_k is new latent vector and vehicle state, P_k is new P_k, K_k is just there for interesting plots lol | X_hat_prior, Y_hat, self.R_k, P_k_prior, Y

        X_klatent_post, P_kl_post, K_kl = self.ukf(latent_hat, sigma_Y_l, self.R_kl, P_xxl.unsqueeze(0), latent_Y) # X_k is new latent vector and vehicle state, P_k is new P_k, K_k is just there for interesting plots lol | X_hat_prior, Y_hat, self.R_k, P_k_prior, Y
      
        return X_kpose_post, X_klatent_post, P_kp_post, P_kl_post, K_kp, K_kl


    def F(self, latent, x_vehicle, P_xx_prior, P_xxl_prior, dt):
        """
        Inputs: Latent features, 6dof Vehicle state , P_x from t-1, P_l from k-1, dt
        Returns: 6dof vehicle time t, latent feature prediction time k, covar x, covar latent 
        """
        latent_hat, xhat_p, P_xhat, P_lhat = self.dynamics(x_vehicle, latent, self.key, dt, P_xx_prior, P_xxl_prior, self.Q_k_l, self.Q_k_x) # takes vehicle state time t, latent space k, current key, dt: returns vehicle state k+1, latent space k+1
        return xhat_p, latent_hat, P_xhat, P_lhat # Pose estimated by latent space forecasting and odom net, Pose estimated by physical dynamics, latent space forecast


    def H(self, p_hat, latent_hat, P_xx, P_xxl):
        # Measurement model H(X_hat) = Y_hat
        P_xx, sigma_p_prior = self.sigma_(p_hat.reshape(1,1,p_hat.shape[2],1), P_xx.squeeze(), self.Q_k_x) # Convert t|t-1 to sigma points
        P_xxl, sigma_l_prior = self.sigma_(latent_hat.unsqueeze(-1), P_xxl, self.Q_k_l) # Convert k|k-1 to sigma points
        
        #yhat_pose_sigma = self.pose(sigma_l_prior) 
        yhat_pose_sigma = sigma_p_prior # self.pose(sigma_l_prior) # takes latent space sigma points at k+1: returns vehicle state sigma K+1
        return yhat_pose_sigma, sigma_l_prior # Yhat at time t+1 based on sigma from latent hat


class encoder_attn(nn.Module):
    def __init__(self):
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

        at_l = attn.AttnLocal
        at_g = attn.AttnGlobal
        at_dyn = attn.AttnDyn

        self.lin_map = nn.Linear(64,256)
        self.q_map = nn.Linear(64,256)
        self.dynamic_map = nn.Linear(8,1)

        self.encode = nn.ModuleList([attn.attention_local_global_layer(dim = dim_encoder, depth = d_encoder, heads = num_heads_encoder, AttnType = at_l)
                                for i in range(layer_depth_encoder)
                                ])

        self.dynamics = nn.ModuleList([attn.attention_local_global_layer(dim = dim_dynamics, depth = d_dynamics, heads = num_heads_dynamics, AttnType = at_dyn)
                                for i in range(layer_depth_dynamics)
                                ])

    def key_emb(self,x):
        """
        The forward takes either an image or a flat vector of arbitrary size, patch embeds it
        then runs it through the corresponding attention network, the norms+avgpools+flattens it again
        """
        skip = x
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