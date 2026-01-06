import abc
import torch
from torch.distributions import Normal, Laplace
import yaml
from math import pi, log
import torch
import torch.nn as nn
class Distribution(abc.ABC):
    """ Potentials abstract class """
    def __init__(self):
        super().__init__()
        # Min
        self.potential_minimizer = None
        self.potential_min = None
        self.keep_minimizer = False # Defaults to False, set to True for rejection sampler/optimization based algs
        pass
    
    def log_prob(self, x):
        # This method calls log_prob and updates the minimizer
        log_dens = self._log_prob(x)
        if self.keep_minimizer:
            xp = x.view((-1,self.dim))
            log_dens_vals = log_dens.view((-1,1))
            argmin = torch.argmin(-log_dens_vals)
            minimum = -log_dens_vals[argmin] 
            
            if self.potential_min is None or minimum < self.potential_min:
                # print(f'Updating Minimizer {xp[argmin]} {minimum}')
                self.potential_min = minimum
                self.potential_minimizer = xp[argmin]  
        return log_dens
    
    def _grad_log_prob(self,x):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            torch.autograd.set_detect_anomaly(True)
            pot = self.log_prob(x)
            return torch.autograd.grad(pot.sum(),x)[0].detach()
    
    def grad_log_prob(self,x):
        return self._grad_log_prob(x)
    
    def gradient(self, x):
        return torch.exp(self.log_prob(x)) * self.grad_log_prob(x)    
import torch
import torch.nn as nn

class EqualBlockIndependentDistribution(Distribution):
    """
    Perfect for:
      - 2D lattice models (block_dim=2, n_blocks = height*width)
      - Sequence of vectors (block_dim=d, n_blocks=T)
      - Fully factorized high-dim distributions
    """
    def __init__(self, n_blocks, block_dim, energy_fn):
        """
        Args:
            n_blocks    : int   → how many independent blocks (G)
            block_dim   : int   → dimension of each block (d_g)
            energy_fn   : callable or nn.Module
                          takes tensor of shape (..., block_dim) 
                          returns energy scalar per sample: (...,)
        """
        super().__init__()
        self.n_blocks  = n_blocks
        self.block_dim = block_dim
        self.dim       = n_blocks * block_dim
        
        self.energy_fn = energy_fn

    def _split_into_blocks(self, x):
        """
        x: (*, n_blocks * block_dim) → (*, n_blocks, block_dim)
        """
        new_shape = list(x.shape[:-1]) + [self.n_blocks, self.block_dim]
        return x.view(*new_shape)

    def _log_prob(self, x):
        """
        Compute log probability: log p(x) = -∑_g V(x_g)
        where V is the energy function.
        """
        blocks = self._split_into_blocks(x)  # (*, n_blocks, block_dim)
        
        # Apply energy function to every block
        if isinstance(self.energy_fn, nn.ModuleList):
            energy_per_block = self.energy_fn[0](blocks)  # (*, n_blocks)
        else:
            energy_per_block = self.energy_fn(blocks)      # (*, n_blocks)
        
        # Sum energy over all blocks
        if energy_per_block.dim() == 1:
            # Single sample: energy_per_block has shape [n_blocks]
            total_energy = energy_per_block.sum()  # scalar
            return -total_energy.unsqueeze(0)       # [1]
        else:
            # Batch: energy_per_block has shape [B, n_blocks]
            total_energy = energy_per_block.sum(dim=-1)  # [B]
            return -total_energy.unsqueeze(-1)            # [B, 1]

    def log_prob_single_dim(self, x_value, dim_idx):
        """
        Evaluate log p(x_dim) for a single dimension.
        
        Since blocks are independent, we need to:
        1. Identify which block this dimension belongs to
        2. Identify which coordinate within that block
        3. Marginalize over other coordinates in that block (if block_dim > 1)
        
        Args:
            x_value: [num_samples] - values for this dimension
            dim_idx: int - which dimension (in flattened space)
            
        Returns:
            log_prob: [num_samples]
        """
        # Determine which block and which coordinate within block
        block_idx = dim_idx // self.block_dim
        coord_in_block = dim_idx % self.block_dim
        
        if self.block_dim == 1:
            # Simple case: each dimension is its own block
            # log p(x_i) = -V(x_i) where V is the energy function
            
            # Reshape to (num_samples, 1) for energy_fn
            x_reshaped = x_value.unsqueeze(-1)  # [num_samples, 1]
            
            if isinstance(self.energy_fn, nn.ModuleList):
                energy = self.energy_fn[0](x_reshaped).squeeze(-1)  # [num_samples]
            else:
                energy = self.energy_fn(x_reshaped).squeeze(-1)      # [num_samples]
            
            return -energy
        
        else:
            # Complex case: need to marginalize over other coordinates in the block
            # This requires numerical integration or sampling
            # For now, raise an error suggesting the user needs to implement this
            raise NotImplementedError(
                f"log_prob_single_dim for block_dim={self.block_dim} > 1 requires "
                f"marginalizing over {self.block_dim - 1} other dimensions in the block. "
                f"This needs to be implemented based on your specific energy function. "
                f"Options: (1) Monte Carlo marginalization, (2) Analytical if energy is "
                f"separable, (3) Use full blocks instead of single dimensions."
            )

    def log_prob_all_dims_batched(self, x):
        """
        VECTORIZED: Evaluate log p(x_i) for ALL dimensions at once.
        
        For independent blocks with block_dim=1, this is straightforward.
        For block_dim>1, this requires marginalization.
        
        Args:
            x: [B, D] where D = n_blocks * block_dim
            
        Returns:
            log_probs: [B, D]
        """
        B = x.shape[0]
        
        if self.block_dim == 1:
            # Simple case: each dimension is its own block
            # log p(x_i) = -V(x_i)
            
            # Reshape to treat each dimension independently
            x_reshaped = x.view(B, self.n_blocks, 1)  # [B, n_blocks, 1]
            
            # Compute energy for each block
            if isinstance(self.energy_fn, nn.ModuleList):
                energy = self.energy_fn[0](x_reshaped)  # [B, n_blocks]
            else:
                energy = self.energy_fn(x_reshaped)     # [B, n_blocks]
            
            # Log prob is negative energy
            log_probs = -energy  # [B, n_blocks]
            
            # Note: output is [B, n_blocks] which equals [B, D] when block_dim=1
            return log_probs
        
        else:
            # Complex case: need to marginalize over other coordinates in each block
            # For now, provide a Monte Carlo approximation
            return self._log_prob_all_dims_batched_mc(x)
    
    def _log_prob_all_dims_batched_mc(self, x, n_mc_samples=1000):
        """
        Monte Carlo approximation of marginal log probabilities.
        
        For dimension i in block g, coordinate c:
        log p(x_i) = log ∫ p(x_g) dx_{g,-c}
                   = log ∫ exp(-V(x_g)) dx_{g,-c}
        
        Args:
            x: [B, D]
            n_mc_samples: number of Monte Carlo samples for marginalization
            
        Returns:
            log_probs: [B, D]
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        log_probs = torch.zeros(B, self.dim, device=device, dtype=dtype)
        
        # Split into blocks
        blocks = self._split_into_blocks(x)  # [B, n_blocks, block_dim]
        
        # For each dimension
        for dim_idx in range(self.dim):
            block_idx = dim_idx // self.block_dim
            coord_in_block = dim_idx % self.block_dim
            
            # Get the observed value for this dimension
            x_observed = blocks[:, block_idx, coord_in_block]  # [B]
            
            # Create MC samples for the other coordinates in this block
            # Shape: [B, n_mc_samples, block_dim]
            mc_blocks = torch.randn(B, n_mc_samples, self.block_dim, 
                                   device=device, dtype=dtype)
            
            # Set the observed coordinate
            mc_blocks[:, :, coord_in_block] = x_observed.unsqueeze(1)
            
            # Compute energies for all MC samples
            # Reshape to [B*n_mc_samples, block_dim] for energy_fn
            mc_blocks_flat = mc_blocks.view(B * n_mc_samples, self.block_dim)
            
            if isinstance(self.energy_fn, nn.ModuleList):
                energies_flat = self.energy_fn[0](mc_blocks_flat)  # [B*n_mc_samples]
            else:
                energies_flat = self.energy_fn(mc_blocks_flat)     # [B*n_mc_samples]
            
            # Reshape back to [B, n_mc_samples]
            energies = energies_flat.view(B, n_mc_samples)
            
            # Compute log p(x_i) using log-sum-exp trick
            # log p(x_i) ≈ log(1/K * sum_k exp(-V_k))
            log_probs[:, dim_idx] = torch.logsumexp(-energies, dim=1) - torch.log(
                torch.tensor(n_mc_samples, dtype=dtype, device=device)
            )
        
        return log_probs

    def _grad_log_prob(self, x):
        blocks = self._split_into_blocks(x).clone()
        blocks.requires_grad_(True)
        
        if isinstance(self.energy_fn, nn.ModuleList):
            energy = self.energy_fn[0](blocks).sum()
        else:
            energy = self.energy_fn(blocks).sum()
            
        grad = torch.autograd.grad(energy, blocks, retain_graph=False)[0]
        grad = grad.view(x.shape)        
        return -grad
    
    def sample(self, tot_samples=1, device='cuda', dtype=torch.float32):
        """
        Sample from the distribution by independently sampling each block.
        Since blocks are independent: p(x₁, ..., x_G) = ∏ p(x_g)
        """
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU instead")
        
        # Initialize ALL blocks for ALL samples at once
        blocks = torch.randn(tot_samples, self.n_blocks, self.block_dim, 
                            device=device, dtype=dtype)
        
        # Run Langevin dynamics on all blocks simultaneously
        n_steps = 500
        step_size = 0.01
        
        for step in range(n_steps):
            blocks.requires_grad_(True)
            
            # Compute energy for ALL blocks at once
            if isinstance(self.energy_fn, nn.ModuleList):
                energy = self.energy_fn[0](blocks)
            else:
                energy = self.energy_fn(blocks)
            
            # Get gradients
            grad = torch.autograd.grad(energy.sum(), blocks)[0]
            blocks = blocks.detach()
            
            # Langevin update for all blocks simultaneously
            noise = torch.randn_like(blocks) * (2 * step_size) ** 0.5
            blocks = blocks - step_size * grad + noise
        
        # Reshape from (tot_samples, n_blocks, block_dim) to (tot_samples, n_blocks * block_dim)
        x = blocks.reshape(tot_samples, -1)
        return x



class ModifiedMueller(Distribution):
    def __init__(self, A, a, b, c, XX, YY):
        super().__init__()
        self.dim = 2
        self.n = 4
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.XX = XX
        self.YY = YY
        self.x_c = -0.033923
        self.y_c = 0.465694      
        self.beta = .1
        self.translation_x = 3.5
        self.translation_y = -6.5
        self.dilatation = 1/5
        
    def transformation(self, xx):
        x = self.dilatation * (xx[:,0] - self.translation_x)
        y = self.dilatation * (xx[:,1] - self.translation_y)
        return x,y
    
    def _log_prob(self, xx):
        new_shape = list(xx.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

        V_m = 0
        for i in range(self.n):
            xi = x- self.XX[i]
            yi = y-self.YY[i]
            V_m+= self.A[i] * torch.exp(self.a[i]* xi**2 \
                    + self.b[i] * xi * yi \
                    + self.c[i] * yi**2)
        V_q = 35.0136 * (x-self.x_c)**2 + 59.8399 * (y-self.y_c)**2
        
        return -self.beta * (V_q + V_m).view(new_shape)
    
    def _grad_log_prob(self, xx):
        curr_shape = list(xx.shape)
        xx = xx.view(-1,self.dim)
        x,y = self.transformation(xx)

        grad_x = 0
        grad_y = 0
        for i in range(self.n):
            xi = x- self.XX[i]
            yi = y-self.YY[i]
            ee = self.A[i] * torch.exp(self.a[i]* xi**2 \
                + self.b[i] * xi * yi \
                + self.c[i] * yi**2)
            grad_x+=  ee * (2 * self.a[i] * xi + self.b[i] * yi)
            grad_y+=  ee * (self.b[i] * xi + 2 * self.c[i] * yi)
        
        # V_q
        grad_x += 2 * 35.0136 * (x-self.x_c)
        grad_y += 2 * 59.8399 * (y-self.y_c)
        grad_x = grad_x.unsqueeze(-1)
        grad_y = grad_y.unsqueeze(-1)
        return -self.beta * torch.cat((grad_x,grad_y),dim=-1).view(curr_shape) * self.dilatation
       
class OneDimensionalGaussian(Distribution):
    # This is a wrapper for Normal
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.dist = Normal(loc=mean, scale=cov**.5)
    
    def sample(self):
        # TODO: Make this in batches
        return self.dist.sample()
    
    def _log_prob(self,x):
        return self.dist.log_prob(x)

    def gradient(self, x):
        dens = torch.exp(self.log_prob(x))
        return - dens * (x - self.mean)/self.cov
class MultivariateGaussian(Distribution):
    def __init__(self, mean, cov):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.Q = torch.linalg.cholesky(self.cov)
        self.inv_cov = torch.linalg.inv(cov)
        self.L = torch.linalg.cholesky(self.inv_cov)
        self.log_det = torch.log(torch.linalg.det(self.cov))
        self.dist = torch.distributions.MultivariateNormal(self.mean, self.cov)
        self.dim = mean.shape[0]
        
        # Pre-compute marginal parameters for each dimension
        self.marginal_means = self.mean  # [D]
        self.marginal_vars = torch.diagonal(self.cov)  # [D] - diagonal elements
        self.marginal_stds = torch.sqrt(self.marginal_vars)  # [D]
    
    def log_prob_single_dim(self, x_value, dim_idx):
        """
        Evaluate log p(x_dim) for a single dimension (marginal).
        
        For a multivariate Gaussian, the marginal of dimension i is:
        p(x_i) = N(x_i | mean_i, cov_ii)
        
        Args:
            x_value: [num_samples] - values for this dimension
            dim_idx: int - which dimension
            
        Returns:
            log_prob: [num_samples]
        """
        mean_i = self.marginal_means[dim_idx]
        std_i = self.marginal_stds[dim_idx]
        
        # Univariate Gaussian log probability
        log_prob = -0.5 * torch.log(2 * torch.tensor(pi)) \
                   - torch.log(std_i) \
                   - 0.5 * ((x_value - mean_i) / std_i) ** 2
        
        return log_prob
    
    def sample(self):
        # TODO: Make this in batches
        return self.Q @ torch.randn_like(self.mean) + self.mean
    
    def _log_prob(self, x):
        new_shape = list(x.shape)
        new_shape[-1] = 1
        new_shape = tuple(new_shape)
        x = x.view((-1, self.dim))
        shift_cov = (self.L.T @ (x - self.mean).T).T
        log_prob = -.5 * (self.dim * log(2 * pi) + self.log_det + torch.sum(shift_cov**2, dim=1)) 
        log_prob = log_prob.view(new_shape)
        return log_prob

    def _grad_log_prob(self, x):
        # This is the gradient of p(x)
        curr_shape = x.shape
        x = x.view((-1, self.dim))
        grad = - (self.inv_cov @ (x - self.mean).T).T
        grad = grad.view(curr_shape)
        return grad
    def log_prob_all_dims_batched(self, x):
        """
        VECTORIZED: Evaluate log p(x_i) for ALL dimensions at once.
        
        Args:
            x: [B, D]
        Returns:
            log_probs: [B, D]
        """
        log_probs = -0.5 * torch.log(2 * torch.tensor(pi, device=x.device)) \
                    - torch.log(self.marginal_stds) \
                    - 0.5 * ((x - self.marginal_means) / self.marginal_stds) ** 2
        return log_probs
class LaplacianDistribution(Distribution):
    def __init__(self, mean, scale):
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.dim = mean.shape[-1]
        self.dist = Laplace(mean,scale)
    
    def sample(self):
        return self.dist.sample()
    
    def _log_prob(self,x):
        return self.dist.log_prob(x).sum(dim=-1,keepdim=True)

class RingDistribution(Distribution):
    # Ring Distribution
    def __init__(self, radius, scale, dim=2):
        super().__init__()
        self.radius = radius
        self.scale = scale
        self.dim = dim
    
    def sample(self,device='cuda',dtype=torch.float32):
        direction = torch.randn((1,self.dim),device=device,dtype=dtype)
        direction = direction/torch.sum(direction**2,-1,keepdim=True)**.5
        radius = self.radius + self.scale * torch.randn(1,device=device,dtype=dtype)
        return direction * radius
    
    def _log_prob(self,x):
        norm = torch.sum(x**2,dim=-1,keepdim=True)**.5
        return - (norm - self.radius)**2/(2 * self.scale**2)
    
class MixtureDistribution(Distribution):
    def __init__(self, c, distributions):
        super().__init__()
        self.n = len(c)
        self.c = c
        self.cats = torch.distributions.Categorical(c)
        self.distributions = distributions
        self.accum = [0.]
        self.dim = self.distributions[0].dim
        
        # Add block_dim attribute (default to 1 for independent dimensions)
        self.block_dim = getattr(distributions[0], 'block_dim', 1)
        
        for i in range(self.n):
            self.accum.append(self.accum[i] + self.c[i].detach().item())
        self.accum = self.accum[1:]

    def _log_prob(self, x):
        log_probs = []
        for i in range(self.n):
            log_probs.append(log(self.c[i]) + self.distributions[i].log_prob(x))
        log_probs = torch.cat(log_probs, dim=-1)
        log_dens = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        return log_dens
    
    def log_prob_single_dim(self, x_value, dim_idx):
        """
        Evaluate marginal log p(x_dim) for a single dimension in a mixture.
        
        For a mixture: p(x_i) = Σ_k c_k * p_k(x_i)
        where p_k(x_i) is the marginal of the k-th component.
        
        Args:
            x_value: [num_samples] - values for this dimension
            dim_idx: int - which dimension
            
        Returns:
            log_prob: [num_samples]
        """
        log_probs = []
        
        for i in range(self.n):
            # Get log prob from each mixture component's marginal
            component_log_prob = self.distributions[i].log_prob_single_dim(x_value, dim_idx)
            # Weight by mixture coefficient
            log_probs.append(torch.log(self.c[i]) + component_log_prob)
        
        # Stack and logsumexp: log(Σ c_k * p_k(x_i))
        log_probs = torch.stack(log_probs, dim=-1)  # [num_samples, n_components]
        log_dens = torch.logsumexp(log_probs, dim=-1)  # [num_samples]
        
        return log_dens
    
    def _grad_log_prob(self, x):
        log_p = self.log_prob(x)
        grad = 0
        for i in range(self.n):
            log_pi = self.distributions[i].log_prob(x)
            grad += self.c[i] * torch.exp(log_pi) * self.distributions[i].grad_log_prob(x)
        return grad / (torch.exp(log_p) + 1e-8)
    
    def sample(self, num_samples):
        one_sample = self.distributions[0].sample()
        samples = torch.zeros(num_samples, self.dim,
                              dtype=one_sample.dtype,
                              device=one_sample.device)
        for i in range(num_samples):
            idx = self.cats.sample()
            samples[i] = self.distributions[idx].sample()
        return samples
    def log_prob_all_dims_batched(self, x):
        """
        VECTORIZED: Evaluate log p(x_i) for ALL dimensions at once for mixture.
        
        Args:
            x: [B, D]
        Returns:
            log_probs: [B, D]
        """
        B, D = x.shape
        
        # Check if all components have the vectorized method
        if all(hasattr(dist, 'log_prob_all_dims_batched') for dist in self.distributions):
            # Fully vectorized path
            log_probs_components = []
            for i in range(self.n):
                component_log_probs = self.distributions[i].log_prob_all_dims_batched(x)
                log_probs_components.append(torch.log(self.c[i]) + component_log_probs)
            
            log_probs_components = torch.stack(log_probs_components, dim=0)  # [n, B, D]
            log_probs = torch.logsumexp(log_probs_components, dim=0)  # [B, D]
            return log_probs
        else:
            # Fall back to per-dimension loop
            log_probs = torch.zeros(B, D, device=x.device)
            for dim_idx in range(D):
                log_probs[:, dim_idx] = self.log_prob_single_dim(x[:, dim_idx], dim_idx)
            return log_probs
class DoubleWell(Distribution):
    def __init__(self,dim, delta):
        super().__init__()
        self.dim = dim
        self.delta = delta
        
    def _log_prob(self, x):
        return - torch.sum((x**2 - self.delta)**2,dim=-1,keepdim=True)

    def _grad_log_prob(self, x):
        return -4 * (x**2 - self.delta) * x
    
class NonContinuousPotential(Distribution):
    # For now just has discontinuities per radius
    def __init__(self, dist : Distribution):
        super().__init__()
        # Radiuses at which we should experience a jump
        self.distribution = dist
        self.dim = dist.dim
        
    def _log_prob(self, x):
        discontinuity = torch.sum(x**2,dim=-1,keepdim=True)**.5
        discontinuity[discontinuity < 5] = 0
        discontinuity[discontinuity > 11] = 0
        discontinuity*=8
        # This helps prevent problems with the backward pass
        return self.distribution._log_prob(x) - discontinuity.floor().detach() 
    
    def _grad_log_prob(self, x):
        return self.distribution._grad_log_prob(x)
    
    def sample(self,num_samples):
        N = num_samples * 100 # TODO: Don't harcode this
        s = self.distribution.sample(N)
        r = torch.rand((N,1),device=s.device)
        acc_prob = torch.exp(self.log_prob(s) - self.distribution.log_prob(s))
        acc_idx = (r < acc_prob).squeeze(-1).bool()
        return s[acc_idx][:num_samples,:]
            
            
            

class DistributionFromPotential(Distribution):
    # This is a wrapper for Normal
    def __init__(self, potential, dim):
        super().__init__()
        self.potential = potential
        self.dim = dim
    
    def _log_prob(self,x):
        return -self.potential(x)

    
def get_distribution(config, device):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.float32)    

    params = yaml.safe_load(open(config.density_parameters_path))

    density = config.density 
    dist = None
    if  density == 'gmm':
        c = to_tensor_type(params['coeffs'])
        means = to_tensor_type(params['means'])
        variances = to_tensor_type(params['variances'])
        n = len(c)
        if config.dimension == 1:
            gaussians = [OneDimensionalGaussian(means[i],variances[i]) for i in range(n)]
        else:
            gaussians = [MultivariateGaussian(means[i],variances[i]) for i in range(n)]

        dist = MixtureDistribution(c, gaussians)
    elif density == 'lmm':
        c = to_tensor_type(params['coeffs'])
        means = to_tensor_type(params['means'])
        scales = to_tensor_type(params['variances'])
        n = len(c)
        laplacians = [LaplacianDistribution(means[i],scales[i]) for i in range(n)]
        
        dist = MixtureDistribution(c,laplacians)
    elif density == 'rmm':
        c = to_tensor_type(params['coeffs'])
        radius = to_tensor_type(params['radius'])
        scales = to_tensor_type(params['variances'])
        n = len(c)
        rings = [RingDistribution(radius[i],scales[i],config.dimension) for i in range(n)]
        dist = MixtureDistribution(c,rings)
    elif density == 'mueller':
        dist = ModifiedMueller(to_tensor_type(params['A']),
                               to_tensor_type(params['a']), 
                               to_tensor_type(params['b']), 
                               to_tensor_type(params['c']),
                               to_tensor_type(params['XX']), 
                               to_tensor_type(params['YY']))
    elif density == 'double-well':
        dist = DoubleWell(config.dimension,3.)
    elif density == 'block_independent':
        n_blocks = params['n_blocks']
        block_dim = params['block_dim']
        energy_type = params.get('energy_type', 'quadratic')  # default to quadratic
        
        # Define the energy function based on type
        if energy_type == 'quadratic':
            # Simple quadratic well: V(x) = 0.5 * ||x||^2
            def energy_fn(blocks):
                # blocks: (..., n_blocks, block_dim) or (..., block_dim)
                return 0.5 * (blocks ** 2).sum(dim=-1)  # (..., n_blocks) or (...)
        
        elif energy_type == 'double_well':
            # Double-well per block: V(x) = (||x||^2 - 1)^2

            def energy_fn(blocks):
                norm_sq = (blocks ** 2).sum(dim=-1)
                return (norm_sq - 1.0) ** 2
        
        elif energy_type == 'neural':
            class MLPEnergy(nn.Module):
                def __init__(self, input_dim, hidden_dims=[15]):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for h_dim in hidden_dims:
                        layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
                        prev_dim = h_dim
                    layers.append(nn.Linear(prev_dim, 1))
                    self.net = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.net(x).squeeze(-1)
            
            hidden_dims = params.get('hidden_dims', [15])
            energy_fn = MLPEnergy(block_dim, hidden_dims)
            energy_fn.to(device)
        
        else:
            raise ValueError(f"Unknown energy_type: {energy_type}")
        
        dist = EqualBlockIndependentDistribution(n_blocks, block_dim, energy_fn)
    else:
        print("Density not implemented yet")
        return
    
    if config.discontinuity:
        dist = NonContinuousPotential(dist)
    
    return dist