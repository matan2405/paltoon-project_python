"""
Lateral Authority Allocator.
VERSION 2.2 - Rate-limited smooth transitions

Changes from V2.1:
- Added rate limiting: max Δλ per step
- Reduced alpha to 0.02 for even smoother response
"""

import numpy as np


class LateralAuthorityAllocator:
    """
    Dynamic authority allocation using Sigmoid function.
    Based on Li et al. 2019.
    
    V2.2: Rate-limited transitions for smooth authority changes.
    """
    
    def __init__(self):
        # Sigmoid parameters
        self.lambda_min = 0.1        # Human dominant (safe situation)
        self.lambda_max = 10.0       # System dominant
        self.force_midpoint = 150.0  # Force [N] for 50-50 split
        self.k_steepness = 0.02      # Sigmoid steepness
        
        # Smoothing parameters
        self.prev_lambda = self.lambda_min
        self.alpha_smoothing = 0.02  # Even slower smoothing (was 0.05)
        
        # Rate limiting - max change per step
        # For dt=0.01s: 0.02/step means ~5 seconds to reach max
        self.max_rate_up = 0.02      # Max increase per dt (λ/step) - was 0.1
        self.max_rate_down = 0.01    # Max decrease per dt (slower release)
        
        print(f"🎯 Authority Allocator V2.2 Initialized")
        print(f"   λ range: {self.lambda_min}-{self.lambda_max}")
        print(f"   Smoothing α: {self.alpha_smoothing}")
        print(f"   Rate limits: +{self.max_rate_up}/step, -{self.max_rate_down}/step")
    
    def compute_authority_ratio(self, risk_force: float, 
                                 lateral_error: float = 0.0,
                                 heading_error: float = 0.0) -> float:
        """
        Compute authority ratio λ(k) with rate limiting.
        """
        force_mag = abs(risk_force)
        
        # 1. Sigmoid-based target from force
        sigmoid = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        target_lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid
        
        # 2. First-order low-pass filter
        filtered_lambda = (self.alpha_smoothing * target_lambda + 
                          (1 - self.alpha_smoothing) * self.prev_lambda)
        
        # 3. Rate limiting
        delta = filtered_lambda - self.prev_lambda
        if delta > 0:
            # Increasing - apply max_rate_up
            delta = min(delta, self.max_rate_up)
        else:
            # Decreasing - apply max_rate_down (slower release)
            delta = max(delta, -self.max_rate_down)
        
        lambda_k = self.prev_lambda + delta
        
        # 4. Clip to valid range
        lambda_k = np.clip(lambda_k, self.lambda_min, self.lambda_max)
        
        # 5. Store for next iteration
        self.prev_lambda = lambda_k
        
        return lambda_k
    
    def get_authority_weights(self, lambda_k: float) -> tuple:
        """Convert λ to weights."""
        total = 1.0 + lambda_k
        return 1.0 / total, lambda_k / total
    
    def reset(self):
        """Reset state for new simulation."""
        self.prev_lambda = self.lambda_min


__all__ = ['LateralAuthorityAllocator']
