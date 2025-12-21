import torch
import utils.densities
from utils.optimizers import nesterovs_minimizer, gradient_descent

def sum_last_dim(x):
    return torch.sum(x,dim=-1, keepdim=True)

def get_rgo_sampling(xk, eta, log_prob, device, threshold, minimizer=None):
    # Sampling from exp(-f(x) - (x-y)^2/2eta)
    num_samples, d = xk.shape #xk is assumed to be [n,d]
    accepted_samples = torch.ones_like(xk) # 1 if rejected 0 if accepted
    potential = lambda x : - log_prob(x)
    w = nesterovs_minimizer(xk, potential, threshold) if \
        minimizer == None else minimizer
    f_eta = potential(w)
    
    proposal = xk + eta **.5 * accepted_samples * torch.randn_like(xk)
    
    exp_h1 = potential(proposal)
    rand_prob = torch.rand((num_samples,1),device=device)
    acc_idx = (accepted_samples * torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1))
    accepted_samples = (~acc_idx).long()
    xk[acc_idx] = proposal[acc_idx]
    return xk, acc_idx

def get_samples(y, eta, distribution : utils.densities.Distribution, num_samples, device,threshold=1e-3):
    # Sampling from potential \prop exp( - f(x) - |x-y|^2/2eta)
    # y = [n,d] outputs [n,num_samples,d]
    n, d = y.shape[0], y.shape[-1]
    yk = y.repeat_interleave(num_samples,dim=0)
    samples, accepted_idx = get_rgo_sampling(yk,eta,distribution.log_prob,device, threshold, minimizer=distribution.potential_minimizer)
    samples = samples.reshape((n, -1, d))
    accepted_idx = accepted_idx.reshape((n,-1,d))
    return samples, accepted_idx
def get_rgo_sampling_partial(x_active, eta, log_prob_single_dim, device, threshold,
                              full_x, active_dims, dim_idx, minimizer=None):
    """
    RGO sampling for a SINGLE dimension (assuming independence).
    
    Args:
        x_active: [num_samples, 1] - current values for ONE dimension
        log_prob_single_dim: function that takes (value, dim_idx) 
        dim_idx: the specific dimension index we're sampling
        ... (other args)
    """
    num_samples = x_active.shape[0]
    
    # Generate proposals for this single dimension
    proposal_active = x_active + (eta ** 0.5) * torch.randn_like(x_active)  # [num_samples, 1]
    
    # Evaluate potential for ONLY this dimension
    potential_single = lambda x_val: -log_prob_single_dim(x_val.squeeze(-1), dim_idx)
    
    # Get minimizer for this dimension only
    if minimizer is None:
        w_active = nesterovs_minimizer(x_active, potential_single, threshold)
    else:
        w_active = minimizer[dim_idx].unsqueeze(0).expand(num_samples, 1)
    
    f_eta = potential_single(w_active).reshape(-1, 1)
    exp_h1 = potential_single(proposal_active).reshape(-1, 1)
    
    # Acceptance test (same as before)
    rand_prob = torch.rand((num_samples, 1), device=device)
    acc_prob = torch.exp(-f_eta) * rand_prob
    acc_idx = (acc_prob <= torch.exp(-exp_h1))
    
    result = torch.where(acc_idx, proposal_active, x_active)
    
    return result, acc_idx.long()


def get_samples_partial(y_active, eta, distribution, num_samples, device,
                        full_x, active_dims, inactive_dims, threshold=1e-3):
    """
    Sample from conditional distribution for active dimensions.
    
    Args:
        y_active: [B, d_active] - current values for active dimensions
        eta: variance parameter  
        distribution: Distribution object with log_prob and potential_minimizer
        num_samples: number of samples to generate per position
        device: torch device
        full_x: [B, D] - full dimensional positions (with inactive dims fixed)
        active_dims: list of active dimension indices
        inactive_dims: list of inactive dimension indices
        threshold: optimization threshold
        
    Returns:
        samples: [B, num_samples, d_active] - proposed values for active dims
        accepted_idx: [B, num_samples, d_active] - binary acceptance indicators
    """
    B, d_active = y_active.shape
    D = full_x.shape[-1]
    
    # Repeat y_active and full_x for num_samples
    y_active_repeated = y_active.repeat_interleave(num_samples, dim=0)  # [B*num_samples, d_active]
    full_x_repeated = full_x.repeat_interleave(num_samples, dim=0)      # [B*num_samples, D]
    
    # Sample using RGO
    samples, acc_idx = get_rgo_sampling_partial(
        y_active_repeated,
        eta,
        distribution.log_prob_single_dim,  # Pass the per-dim function
        device,
        threshold,
        full_x_repeated,
        active_dims,
        dim_idx=active_dims[0],  # Since you're doing 1 dim at a time
        minimizer=distribution.potential_minimizer
    )
    
    # Reshape back to [B, num_samples, d_active]
    samples = samples.reshape(B, num_samples, d_active)
    acc_idx = acc_idx.reshape(B, num_samples, d_active)
    
    return samples, acc_idx
def get_samples_independent_blocks(x_blocks, eta, distribution, num_samples, device, active_dims):
    """
    Sample independent blocks for EqualBlockIndependentDistribution.
    
    Args:
        x_blocks: [B, num_active] current values for active dimensions
        eta: variance parameter
        distribution: EqualBlockIndependentDistribution instance
        num_samples: number of samples per dimension
        device: torch device
        active_dims: tensor of active dimension indices
        
    Returns:
        proposals: [B, num_samples, num_active]
        accepted: [B, num_samples, num_active]
    """
    B, num_active = x_blocks.shape
    block_dim = distribution.block_dim
    
    # Generate proposals
    proposals = x_blocks.unsqueeze(1) + (eta ** 0.5) * torch.randn(
        B, num_samples, num_active, device=device
    )  # [B, num_samples, num_active]
    
    # For block_dim=1: each dimension is one block
    # For block_dim>1: need to handle carefully
    
    if block_dim == 1:
        # Reshape proposals to match energy_fn input format: (batch, n_blocks, block_dim)
        # We want to evaluate each (sample, dimension) pair as a separate block
        # Shape: [B*num_samples, num_active, 1]
        proposals_for_energy = proposals.transpose(1, 2).reshape(B * num_samples, num_active, 1)
        
        # Evaluate energy for all blocks at once
        energy_per_block = distribution.energy_fn(proposals_for_energy)
        
        # Reshape back: [B, num_samples, num_active]
        energy_per_block = energy_per_block.reshape(B, num_samples, num_active)
        
        # Log probability per block (negative energy)
        log_p_proposal = -energy_per_block  # [B, num_samples, num_active]
        
        # Get mode energy
        # The mode is stored as full vector, extract active blocks
        mode_blocks = distribution.potential_minimizer[active_dims].unsqueeze(0).unsqueeze(-1)  # [1, num_active, 1]
        
        mode_energy = distribution.energy_fn(mode_blocks)
        
        log_p_mode = -mode_energy  # [1, num_active]
        
    else:
        # block_dim > 1: active_dims should correspond to complete blocks
        # This is more complex - need to ensure active_dims aligns with block boundaries
        raise NotImplementedError(f"block_dim={block_dim} > 1 not yet supported")
    
    # Acceptance probability: min(1, p(proposal) / p(mode))
    log_acc_prob = log_p_proposal - log_p_mode  # [B, num_samples, num_active]
    acc_probs = torch.exp(torch.clamp(log_acc_prob, max=0))
    
    # Accept/reject
    rand_prob = torch.rand((B, num_samples, num_active), device=device)
    accepted = (rand_prob <= acc_probs).long()
    
    return proposals, accepted