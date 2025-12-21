import torch
from collections import deque

class AdaptiveProposalCache:
    """Stores accepted samples and constructs adaptive proposal distributions"""
    
    def __init__(self, dim, device, max_cache_size=100, num_components=5, 
                 sample_fraction=0.1, **kwargs):
        self.dim = dim
        self.device = device
        self.max_cache_size = max_cache_size
        self.num_components = num_components
        self.sample_fraction = sample_fraction  # Only cache a fraction of samples
        
        # Cache for different time/scaling values
        self.cache = {}  # key: (tt_rounded, scaling_rounded), value: samples
        self.component_params = {}  # GMM parameters per time step
        
    def add_samples(self, tt, scaling, samples, weights=None):
        """Add accepted samples to cache for specific time/scaling"""
        # print(f"[Cache] add_samples called: {len(samples)} samples")
        key = self._get_key(tt, scaling)
        
        if key not in self.cache:
            self.cache[key] = deque(maxlen=self.max_cache_size)
        
        # Only subsample a fraction of samples to reduce overhead
        num_to_add = max(1, int(len(samples) * self.sample_fraction))
        # print(f"[Cache] Will add {num_to_add} samples to cache")
        
        # Randomly select samples to cache
        if len(samples) > num_to_add:
            indices = torch.randperm(len(samples))[:num_to_add]
            samples_to_add = samples[indices]
        else:
            samples_to_add = samples
        
        # Add samples with optional importance weights
        for i in range(len(samples_to_add)):
            sample = samples_to_add[i] if samples_to_add.dim() == 2 else samples_to_add[i].reshape(-1)
            weight = weights[i] if weights is not None else 1.0
            self.cache[key].append((sample.detach().clone(), float(weight)))
        
        # print(f"[Cache] Cache now has {len(self.cache[key])} samples for key {key}")
        
        # Update GMM components if we have enough samples (lower threshold)
        if len(self.cache[key]) >= 50:
            # print(f"[Cache] Triggering GMM fit with {len(self.cache[key])} samples")
            self._fit_gmm(key)
            # print(f"[Cache] GMM fit complete")
    
    def _get_key(self, tt, scaling):
        """Round time/scaling to create cache keys"""
        tt_val = float(tt.item()) if torch.is_tensor(tt) else float(tt)
        scaling_val = float(scaling.item()) if torch.is_tensor(scaling) else float(scaling)
        return (round(tt_val, 2), round(scaling_val, 2))
    
    def _fit_gmm(self, key):
        """Fit Gaussian Mixture Model to cached samples"""
        # print(f"[GMM] Starting _fit_gmm for key {key}")
        samples_list = [s for s, w in self.cache[key]]
        weights_list = [w for s, w in self.cache[key]]
        
        # print(f"[GMM] Extracted {len(samples_list)} samples")
        
        if len(samples_list) < self.num_components:
            # print(f"[GMM] Too few samples ({len(samples_list)} < {self.num_components}), skipping")
            return
        
        # print(f"[GMM] Stacking samples...")
        samples = torch.stack(samples_list)
        weights = torch.tensor(weights_list, device=self.device)
        weights = weights / weights.sum()
        

        # Use K-means++ initialization for GMM components
        means, covs, mix_weights = self._kmeans_plus_plus_gmm(
            samples, weights, self.num_components
        )

        
        self.component_params[key] = {
            'means': means,
            'covs': covs,
            'mix_weights': mix_weights
        }

    
    def _kmeans_plus_plus_gmm(self, samples, weights, k):
        """Initialize GMM using weighted k-means++ - FAST VERSION"""
        n = samples.shape[0]
        
        # Early exit for small sample sizes
        if n < k:
            k = n
        
        # Choose first center randomly (weighted)
        idx = torch.multinomial(weights, 1)
        centers = [samples[idx]]
        
        # Choose remaining centers (simplified for speed)
        for _ in range(k - 1):
            dists = torch.stack([
                torch.sum((samples - c)**2, dim=-1) 
                for c in centers
            ]).min(dim=0)[0]
            
            probs = weights * dists
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            centers.append(samples[idx])
        
        centers = torch.cat(centers, dim=0)
        
        # Simplified assignment (no full k-means iterations)
        dists_all = torch.cdist(samples, centers)
        assignments = torch.argmin(dists_all, dim=1)
        
        # Compute means and covariances
        means = []
        covs = []
        mix_weights = []
        
        for c in range(k):
            mask = (assignments == c)
            if mask.sum() == 0:
                means.append(samples[torch.randint(0, n, (1,))])
                covs.append(torch.eye(self.dim, device=self.device))
                mix_weights.append(1.0 / k)
            else:
                cluster_samples = samples[mask]
                cluster_weights = weights[mask]
                cluster_weights = cluster_weights / cluster_weights.sum()
                
                mean = (cluster_samples * cluster_weights.unsqueeze(-1)).sum(dim=0)
                centered = cluster_samples - mean
                
                # Simplified covariance (diagonal + regularization)
                var = (centered**2 * cluster_weights.unsqueeze(-1)).sum(dim=0)
                cov = torch.diag(var + 1e-3)  # Diagonal covariance for speed
                
                means.append(mean)
                covs.append(cov)
                mix_weights.append(mask.sum().float() / n)
        
        return (torch.stack(means), torch.stack(covs), 
                torch.tensor(mix_weights, device=self.device))
    
    def has_cached_params(self, eta):
        """Check if we have learned parameters for similar eta"""
        key = self._find_nearest_key(eta)
        return key in self.component_params if key is not None else False
    
    def _find_nearest_key(self, eta):
        """Find cached parameters for nearest time step"""
        if not self.component_params:
            return None
        
        eta_val = float(eta.item()) if torch.is_tensor(eta) else float(eta)
        keys = list(self.component_params.keys())
        dists = [abs(k[1] - eta_val) for k in keys]
        nearest_idx = torch.tensor(dists).argmin()
        return keys[nearest_idx]
    
    def get_proposal_params(self, eta):
        """Get GMM parameters for sampling"""
        key = self._find_nearest_key(eta)
        if key and key in self.component_params:
            return self.component_params[key]
        return None
    
    def sample_from_gmm(self, y, eta, params, num_samples):
        """Sample from Gaussian Mixture Model - FULLY VECTORIZED VERSION"""
        n, d = y.shape
        
        means = params['means']  # [k, d]
        covs = params['covs']    # [k, d, d]  (diagonal matrices)
        mix_weights = params['mix_weights']  # [k]
        
        # Sample component assignments for ALL samples at once
        total_samples = n * num_samples
        component_indices = torch.multinomial(
            mix_weights.repeat(total_samples, 1), 1
        ).squeeze()
        
        # Handle edge case where only 1 sample
        if component_indices.dim() == 0:
            component_indices = component_indices.unsqueeze(0)
        
        component_indices = component_indices.reshape(n, num_samples)
        
        # Get selected means and variances (vectorized!)
        selected_means = means[component_indices]  # [n, num_samples, d]
        
        # Extract diagonal elements from covariance matrices
        variances = torch.stack([torch.diag(cov) for cov in covs])  # [k, d]
        selected_vars = variances[component_indices]  # [n, num_samples, d]
        
        # Blend with current positions
        y_expanded = y.unsqueeze(1).expand(n, num_samples, d)  # [n, num_samples, d]
        blended_means = 0.7 * selected_means + 0.3 * y_expanded
        blended_stds = (selected_vars + eta) ** 0.5
        
        # Sample all at once (no loops!)
        z = torch.randn(n, num_samples, d, device=self.device)
        samples = blended_means + blended_stds * z
        
        return samples