#!/usr/bin/env python3
"""
File: authority_allocator.py
Description: Contains the dynamic authority allocation logic based on driving risk.
"""

import numpy as np

class ImprovedAuthorityAllocator:
    """Enhanced authority allocation based on Li et al. Table 2"""
    
    def __init__(self):
        # Li et al. Table 2 - Force ranges and lambda values
         # Original paper: Force=0 → ln(λ)=-1.78 → λ=0.169 (Human dominant!)
        self.force_ranges = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
        
        # Convert from ln(λ) to λ values from Table 2
        ln_lambda = np.array([-1.78, -1.42, -0.83, -0.42, 0.00, 0.21, 0.63, 1.17, 1.61, 1.83, 2.21, 2.56, 3.34, 4.23, 4.67])
        self.lambda_values = np.exp(ln_lambda)

        # Smoothing parameters - ENHANCED for stability
        self.prev_lambda = 0.169  # Start with human dominant
        self.smoothing_alpha = 0.8  # INCREASED from 0.7 to 0.8 for more smoothing
        self.max_change_rate = 1.5  # REDUCED from 2.0 to 1.5 for smoother transitions
        
    def compute_authority_ratio(self, field_force: np.ndarray) -> float:
        """Compute λ(k) based on field force magnitude"""
        
        force_mag = np.linalg.norm(field_force)
        
        # IMPROVED: Enhanced lookup with interpolation
        lambda_k = np.interp(force_mag, self.force_ranges, self.lambda_values)
                        #    np.append(self.lambda_values, 5.0))  # REDUCED max from 100 to 5
        
        # ENHANCED: Adaptive response based on force direction
        if len(field_force) >= 2:
            lateral_component = abs(field_force[1])
            if lateral_component > 200.0:  # High lateral risk
                lambda_k *= 1.2  # REDUCED multiplier from 1.5 to 1.2
                
        # Rate limiting with increased responsiveness
        max_increase = self.prev_lambda * self.max_change_rate
        min_decrease = self.prev_lambda / (self.max_change_rate)
        lambda_k = np.clip(lambda_k, min_decrease, max_increase)
        
        # Smoothing with better responsiveness
        alpha = self.smoothing_alpha
        if lambda_k > self.prev_lambda * 1.2:  # Emergency increase
            alpha = 0.8  # INCREASED from 0.7 to 0.8 for even smoother emergency response
        elif lambda_k < self.prev_lambda * 0.8:
            alpha = 0.9
        
        lambda_k = alpha * lambda_k + (1 - alpha) * self.prev_lambda
        
        # Bounds - REDUCED upper bound for stability
        lambda_k = np.clip(lambda_k, 0.05, 50.0)
        
        self.prev_lambda = lambda_k
        return lambda_k
