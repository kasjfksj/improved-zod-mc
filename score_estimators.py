import torch
import abc

from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers
import samplers.ula as ula
from samplers.adaptive_proposal import AdaptiveProposalCache
import time


class ScoreEstimator(abc.ABC):
    def __init__(self, dist: Distribution,
                 sde, device,def_num_batches=1,
                 def_num_samples=10000) -> None:
        self.sde = sde
        self.dist = dist
        self.device = device
        self.default_num_batches = def_num_batches
        self.default_num_samples = def_num_samples
        self.dim = self.dist.dim

    @abc.abstractmethod
    def score_estimator(self, x,tt, num_batches=None, num_rej_samples=None):
        pass

import time
import torch

class PartialZODMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist, sde, device,
                 active_block_size=1,
                 def_num_batches=1,
                 def_num_rej_samples=10000,
                 max_iters_opt=10,  # ← Reduced from 50
                 cycle_mode='sequential',
                 debug=True,
                 profile=True):
        
        super().__init__(dist, sde, device, def_num_batches, def_num_rej_samples)
        self.active_block_size = min(active_block_size, dist.dim)
        self.cycle_mode = cycle_mode
        self.debug = debug
        self.profile = profile
        self.num_blocks = (dist.dim + active_block_size - 1) // active_block_size

        if cycle_mode == 'sequential':
            self.dim_blocks = []
            for i in range(self.num_blocks):
                start = i * active_block_size
                end = min(start + active_block_size, dist.dim)
                self.dim_blocks.append(list(range(start, end)))
        
        minimizer_start = time.time()
        
        dist.keep_minimizer = True
        dist.potential_minimizer = torch.zeros(dist.dim, device=device)
        dist.f_eta_minimizers = torch.zeros(dist.dim, device=device)
        
        # Try to use GMM mean as minimizer (much faster!)
        if hasattr(dist, 'distributions') and hasattr(dist, 'c'):
            # For mixture distribution, use weighted mean
            weighted_mean = torch.zeros(dist.dim, device=device)
            for i, (coeff, component) in enumerate(zip(dist.c, dist.distributions)):
                if hasattr(component, 'mean'):
                    weighted_mean += coeff * component.mean
            dist.potential_minimizer = weighted_mean
        else:
            # Fall back to optimization (slower)
            for dim_idx in range(dist.dim):
                def potential_single_dim(x_val):
                    if x_val.dim() == 0:
                        x_val = x_val.unsqueeze(0)
                    return -dist.log_prob_single_dim(x_val, dim_idx)
                
                x_init = torch.randn(1, device=device) * 2
                try:
                    minimizer_dim = optimizers.newton_conjugate_gradient(
                        x_init, potential_single_dim, max_iters_opt
                    )
                    dist.potential_minimizer[dim_idx] = minimizer_dim.item()
                except:
                    dist.potential_minimizer[dim_idx] = 0.0
        
        # Pre-compute f_eta for all minimizers (one-time cost)
        for dim_idx in range(dist.dim):
            dist.f_eta_minimizers[dim_idx] = -dist.log_prob_single_dim(
                torch.tensor([dist.potential_minimizer[dim_idx]], device=device),
                dim_idx
            ).item()
        
        minimizer_time = time.time() - minimizer_start
        if self.debug:
            print(f"[INIT] Minimizer computation took: {minimizer_time:.4f}s")

    def score_estimator(self, x, tt, num_batches=None, num_samples=None):
        scaling = self.sde.scaling(tt)
        variance_conv = (1 / scaling)**2 - 1
        num_batches = self.default_num_batches if num_batches is None else num_batches
        num_samples = self.default_num_samples if num_samples is None else num_samples
        
        if self.profile:
            score, times = self._zodmc_all_dimensions_batched_optimized(
                x, scaling, variance_conv, num_batches, num_samples, x.device, profile=True
            )
            
            print("\n" + "="*60)
            print("TIME BREAKDOWN (P-ZOD-MC OPTIMIZED)")
            print("="*60)
            for key, val in times.items():
                pct = 100 * val / times['total'] if times['total'] > 0 else 0
                print(f"{key:25s}: {val:8.4f}s ({pct:5.1f}%)")
            print("="*60 + "\n")
        else:
            score = self._zodmc_all_dimensions_batched_optimized(
                x, scaling, variance_conv, num_batches, num_samples, x.device, profile=False
            )
        
        return score

    def _zodmc_all_dimensions_batched_optimized(self, x, scaling, variance_conv, 
                                                 num_batches, num_samples, device, profile=False):
        """Fully optimized version with vectorization"""
        times = {} if profile else None
        total_start = time.time() if profile else None
        
        B, D = x.shape
        x_scaled = x / scaling if scaling.dim() == 0 else x / scaling.unsqueeze(-1)
        
        mean_estimate = torch.zeros(B, D, device=device)
        num_good_samples = torch.zeros(B, D, device=device)
        
        # Pre-computed values (broadcast ready)
        f_eta_minimizers = self.dist.f_eta_minimizers.view(1, 1, D)  # [1, 1, D]
        
        if profile:
            times['proposal_gen'] = 0
            times['log_prob_eval'] = 0
            times['acceptance_test'] = 0
            times['statistics_update'] = 0
        
        for batch_idx in range(num_batches):
            # Proposal generation
            if profile: t1 = time.time()
            proposals = x_scaled.unsqueeze(1) + (variance_conv ** 0.5) * torch.randn(
                B, num_samples, D, device=device
            )
            if profile: times['proposal_gen'] += time.time() - t1
            
            # OPTIMIZED: Vectorized log probability evaluation
            if profile: t2 = time.time()
            proposals_flat = proposals.reshape(B * num_samples, D)
            
            # Check if dist has vectorized method
            if hasattr(self.dist, 'log_prob_all_dims_batched'):
                log_probs_all = self.dist.log_prob_all_dims_batched(proposals_flat)
            else:
                # Fall back to loop (slower)
                log_probs_all = torch.zeros(B * num_samples, D, device=device)
                for dim_idx in range(D):
                    log_probs_all[:, dim_idx] = self.dist.log_prob_single_dim(
                        proposals_flat[:, dim_idx], dim_idx
                    )
            
            log_probs = log_probs_all.reshape(B, num_samples, D)
            if profile: times['log_prob_eval'] += time.time() - t2
            
            # OPTIMIZED: Vectorized acceptance test in LOG SPACE (no loop!)
            if profile: t3 = time.time()
            f_eta = f_eta_minimizers.expand(B, 1, D)  # Broadcast pre-computed values
            
            # RGO acceptance: U <= exp(f_eta - exp_h1) where exp_h1 = -log_prob
            # Equivalently: log(U) <= f_eta - exp_h1
            # Equivalently: log(U) <= f_eta + log_prob
            rand_prob = torch.rand(B, num_samples, D, device=device)
            log_rand = torch.log(rand_prob + 1e-10)  # Add epsilon for stability
            acc_idx = (log_rand <= f_eta + log_probs).float()
            if profile: times['acceptance_test'] += time.time() - t3
            
            # Statistics update
            if profile: t4 = time.time()
            num_good_samples += acc_idx.sum(dim=1)
            mean_estimate += (proposals * acc_idx).sum(dim=1)
            if profile: times['statistics_update'] += time.time() - t4
        
        # Final score computation
        if profile: t5 = time.time()
        num_good_samples[num_good_samples == 0] = 1
        mean_estimate = mean_estimate / num_good_samples
        score = (scaling * mean_estimate - x) / (1 - scaling**2 + 1e-8)
        if profile: 
            times['final_computation'] = time.time() - t5
            times['total'] = time.time() - total_start
        
        if profile:
            return score, times
        return score



class ZODMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_rej_samples=10000,
                 max_iters_opt=50
                 ) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_rej_samples)
        
        # Set up distribution correctly
        dist.keep_minimizer = True
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dist.dim,device=device),
                                                         lambda x : -self.dist.log_prob(x), 
                                                         max_iters_opt)
        dist.log_prob(minimizer) # To make sure we update with the right minimizer
    
    def score_estimator(self, x,tt, num_batches=None, num_samples=None):
        scaling = self.sde.scaling(tt)
        variance_conv = (1/scaling)**2 - 1
        score_estimate = torch.zeros_like(x)
        num_batches = self.default_num_batches if num_batches is None else num_batches
        num_samples = self.default_num_samples if num_samples is None else num_samples

        assert num_batches > 0 and num_samples > 0, 'Number of samples needs to be a positive integer'
        
        mean_estimate = 0
        num_good_samples = torch.zeros((x.shape[0],1),device=self.device)
        import time
        t = time.time()
        for _ in range(num_batches):
            samples_from_p0t, acc_idx = rejection_sampler.get_samples(x/scaling, variance_conv,
                                                                                    self.dist,
                                                                                    num_samples, 
                                                                                    self.device)
            num_good_samples += torch.sum(acc_idx, dim=(1,2)).unsqueeze(-1).to(torch.double)/self.dim
            mean_estimate += torch.sum(samples_from_p0t * acc_idx,dim=1)
        num_good_samples[num_good_samples == 0] += 1
        # You can print this if you want to know how many samples are getting accepted per time step
        # print(f'{tt.item() : .5f} {num_good_samples.mean() : .2f}') 
        mean_estimate /= num_good_samples
        print(time.time() - t )
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate


class RDMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 initial_cond_normal=True) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal
        
    def score_estimator(self, x,tt):
        scaling = self.sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = self.default_num_samples
        score_estimate = torch.zeros_like(x)
        big_x = x.repeat_interleave(num_samples,dim=0)
        def grad_log_prob_0t(x0):
            return self.dist.grad_log_prob(x0) + scaling * (big_x - scaling * x0) / (1 - scaling ** 2)
        
        mean_estimate = 0
        x0 = big_x

        for _ in range(self.default_num_batches):
            if self.initial_cond_normal:
                x0 = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
            samples_from_p0t = ula.get_ula_samples(x0,grad_log_prob_0t,self.ula_step_size,self.ula_steps)
            samples_from_p0t = samples_from_p0t.view((-1,num_samples, self.dim))
            
            mean_estimate += torch.sum(samples_from_p0t, dim = 1)
        mean_estimate/= (self.default_num_batches * self.default_num_samples)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

class RSDMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 num_recursive_steps=3,
                 initial_cond_normal=True) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal
        self.num_recursive_steps = num_recursive_steps
        
    def _recursive_langevin(self, x,tt,k=None):
        if k is None:
            k = self.num_recursive_steps
        if k == 0 or tt < .2:
            return self.dist.grad_log_prob(x)
        
        num_samples = self.default_num_samples
        scaling = self.sde.scaling(tt)
        # inv_scaling = 1/scaling
        h = self.ula_step_size      

        big_x = x.repeat_interleave(num_samples,dim=0) 
        x0 = big_x.detach().clone()   
        # x0 = inv_scaling * x0 + torch.randn_like(x0) * (inv_scaling**2 -1)  # q0 initialization
        for _ in range(self.ula_steps):
            score = self._recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,self.dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate
    
    def score_estimator(self, x, tt):
        
        score_estimate = 0
        for _ in range(self.default_num_batches):
            score_estimate+= self._recursive_langevin(x,tt,self.num_recursive_steps)
        score_estimate/= self.default_num_batches
        return score_estimate
        

def get_score_function(config, dist : Distribution, sde, device):
    """
        The following method returns a method that approximates the score
    """
    grad_logdensity = dist.grad_log_prob
    dim = dist.dim

    
    def get_recursive_langevin(x,tt,k=config.num_recursive_steps):
        if k == 0 or tt < .2:
            return grad_logdensity(x)
        
        num_samples = config.num_estimator_samples
        scaling = sde.scaling(tt)
        # inv_scaling = 1/scaling
        h = config.ula_step_size      

        big_x = x.repeat_interleave(num_samples,dim=0) 
        x0 = big_x.detach().clone()   
        # x0 = inv_scaling * x0 + torch.randn_like(x0) * (inv_scaling**2 -1)  # q0 initialization
        for _ in range(config.num_sampler_iterations):
            score = get_recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

        
    if config.score_method == 'p0t' and config.p0t_method == 'rejection' and config.use_partial_zodmc:

        return PartialZODMC_ScoreEstimator(dist,sde,device,active_block_size=config.active_dim,
                                    def_num_batches=config.num_estimator_batches,
                                    def_num_rej_samples=config.num_estimator_samples).score_estimator
    elif config.score_method == 'p0t' and config.p0t_method == 'rejection':
        return ZODMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_rej_samples=config.num_estimator_samples).score_estimator
    elif config.score_method == 'p0t' and config.p0t_method == 'ula':
        initial_cond_normal= True if config.rdmc_initial_condition.lower() == 'normal' else False
        return RDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                ula_steps=config.num_sampler_iterations,
                                initial_cond_normal=initial_cond_normal).score_estimator
    elif config.score_method == 'recursive':
        return RSDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                num_recursive_steps=config.num_recursive_steps,
                                ula_steps=config.num_sampler_iterations).score_estimator
    