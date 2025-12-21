import torch
import utils.densities
from utils.optimizers import nesterovs_minimizer

def sum_last_dim(x):
    return torch.sum(x, dim=-1, keepdim=True)


def get_rgo_sampling(x_active, eta, log_prob, device, threshold, 
                     full_x, active_dim_idx, inactive_dims, minimizer=None):
    """
    RGO sampling for a single active dimension while keeping inactive dims fixed.
    
    Args:
        x_active: Current samples for active dimension [n, 1]
        eta: Variance parameter
        log_prob: Log probability function (takes full-dimensional input)
        device: Torch device
        threshold: Convergence threshold
        full_x: Full-dimensional context [n, D]
        active_dim_idx: Index of the active dimension being sampled
        inactive_dims: Tensor of inactive dimension indices
        minimizer: Pre-computed minimizer (optional)
    
    Returns:
        samples: Updated samples [n, 1]
        accepted_idx: Acceptance indicators [n, 1]
    """
    num_samples = x_active.shape[0]
    potential = lambda x_full: -log_prob(x_full)
    
    # Create full-dimensional proposal
    proposal_full = full_x.clone()
    proposal_active = x_active + (eta ** 0.5) * torch.randn_like(x_active)
    proposal_full[:, active_dim_idx] = proposal_active.squeeze(-1)
    
    # Find minimizer
    if minimizer is None:
        w_full = full_x.clone()
        
        def active_potential(x_val):
            temp = full_x.clone()
            temp[:, active_dim_idx] = x_val.squeeze(-1)
            return potential(temp)
        
        w_active = nesterovs_minimizer(x_active, active_potential, threshold)
        w_full[:, active_dim_idx] = w_active.squeeze(-1)
    else:
        w_full = minimizer
    
    # Compute potentials
    f_eta = potential(w_full)
    exp_h1 = potential(proposal_full)
    
    # Acceptance/rejection
    rand_prob = torch.rand((num_samples, 1), device=device)
    acc_idx = (torch.exp(-f_eta) * rand_prob <= torch.exp(-exp_h1))
    
    # Update samples
    result = torch.where(acc_idx, proposal_active, x_active)
    
    return result, acc_idx


def get_samples(y, eta, distribution, num_samples, device, 
                full_x=None, active_dims=None, inactive_dims=None, threshold=1e-3):
    """
    Generate samples using rejection sampling for specified active dimensions.
    
    Args:
        y: Input samples for active dimension(s) [n, d_active]
        eta: Variance parameter
        distribution: Distribution object with log_prob method
        num_samples: Number of samples to generate per input
        device: Torch device
        full_x: Full-dimensional context [n, D] (required for partial sampling)
        active_dims: Tensor of active dimension indices (required for partial sampling)
        inactive_dims: Tensor of inactive dimension indices
        threshold: Optimizer convergence threshold
    
    Returns:
        samples: [n, num_samples, d_active]
        accepted_idx: [n, num_samples, d_active]
    """
    n, d_active = y.shape[0], y.shape[-1]
    
    # Replicate inputs
    yk = y.repeat_interleave(num_samples, dim=0)  # [n*num_samples, d_active]
    
    # For partial sampling, we need full_x
    if full_x is not None and active_dims is not None:
        full_xk = full_x.repeat_interleave(num_samples, dim=0)  # [n*num_samples, D]
        active_dim_idx = active_dims[0].item() if len(active_dims) == 1 else active_dims[0]
        
        # Get minimizer if available
        minimizer = getattr(distribution, 'potential_minimizer', None)
        if minimizer is not None:
            minimizer = minimizer.repeat_interleave(num_samples, dim=0)
        
        samples, accepted_idx = get_rgo_sampling(
            yk, eta, distribution.log_prob, device, threshold,
            full_xk, active_dim_idx, inactive_dims, minimizer
        )
    else:
        # Fallback: original full-dimensional sampling
        raise NotImplementedError("Full-dimensional sampling not implemented in this version")
    
    # Reshape outputs
    samples = samples.reshape((n, -1, d_active))
    accepted_idx = accepted_idx.reshape((n, -1, d_active))
    
    return samples, accepted_idx