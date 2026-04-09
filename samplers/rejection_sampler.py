import torch
import utils.densities
from utils.optimizers import nesterovs_minimizer, gradient_descent

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, eta, log_prob, device, threshold, minimizer=None):
    num_samples, d = xk.shape
    potential = lambda x: -log_prob(x)
    
    w = nesterovs_minimizer(xk, potential, threshold) if minimizer is None else minimizer
    f_eta = potential(w)
    
    proposals = xk + (eta ** 0.5) * torch.randn_like(xk)
    exp_h1 = potential(proposals)
    
    rand_prob = torch.rand((num_samples, 1), device=device)
    acc_mask = (torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1))  # [n, 1]
    acc_mask = acc_mask.expand(-1, d)                                  # [n, d]
    
    xk = torch.where(acc_mask, proposals, xk)
    return xk, acc_mask

def get_samples(y, eta, distribution: utils.densities.Distribution, num_samples, device, threshold=1e-3):
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat_interleave(num_samples, dim=0)
    samples, accepted_idx = get_rgo_sampling(yk, eta, distribution.log_prob, device, threshold,
                                             minimizer=distribution.potential_minimizer)
    samples = samples.reshape((n, -1, d))
    accepted_idx = accepted_idx.reshape((n, -1, d))
    return samples, accepted_idx

def get_rgo_sampling_partial(xk, eta, log_prob, device, threshold, minimizer=None):
    num_samples, d = xk.shape
    potential = lambda x: -log_prob(x)
    w = nesterovs_minimizer(xk, potential, threshold) if minimizer is None else minimizer
    f_eta = potential(w)                                   
    proposals = xk + (eta ** 0.5) * torch.randn_like(xk)  
    exp_h1 = potential(proposals)                                     
    rand_prob = torch.rand(num_samples, d, device=device)      
    acc_mask = (torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1)) 
    xk = torch.where(acc_mask, proposals, xk)                       
    return xk, acc_mask

def get_samples_partial(y, eta, distribution, num_samples, device, threshold=1e-3):
    n, d = y.shape
    yk = y.repeat_interleave(num_samples, dim=0)
    samples, accepted_idx = get_rgo_sampling_partial(yk, eta, distribution.log_prob, device, threshold,
                                                     minimizer=distribution.potential_minimizer)
    samples = samples.reshape((n, -1, d))
    accepted_idx = accepted_idx.reshape((n, -1, d))
    return samples, accepted_idx