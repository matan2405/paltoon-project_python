#!/usr/bin/env python3
"""
File: longitudinal_authority_allocator.py
Description: Dynamic authority allocation based on Safety and Performance.

VERSION 4.0 - IMPROVED GAP CLOSING
==================================
Key changes from V3:
1. Lower threshold for gap-based authority (2m instead of 10m)
2. Both positive AND negative gap errors activate system authority
3. Higher gain for catching up (positive gap error)
"""

import numpy as np


class LongitudinalAuthorityAllocator:
    """
    Dynamic authority allocation based on a continuous Sigmoid function.
    Updated: More aggressive gap closing for BOTH large and small errors.
    """

    def __init__(self):
        # Sigmoid Parameters for SAFETY (Risk based)
        self.lambda_min = 0.1   # Human dominant when safe
        self.lambda_max = 100.0 # System dominant when dangerous
        self.force_midpoint = 400.0 
        self.k_steepness = 0.015

        self.prev_lambda = self.lambda_min
        self.alpha_smoothing = 0.05 
        
        # Store last computed authority values for logging
        self.last_lambda_safety = 0.0
        self.last_lambda_performance = 0.0

        print(f"🛡️ Authority Allocator V4 (Improved Gap Closing) Initialized")

    def compute_authority_ratio(self, risk_force: float, gap_error: float = 0.0, velocity_error: float = 0.0) -> float:
        """
        Computes authority ratio lambda(k).
        Logic: Max(Safety_Need, Performance_Need)
        
        V4 CHANGES:
        - Lower threshold: 2m instead of 10m
        - Handles both positive (too far) and negative (too close) errors
        - Progressive gain based on magnitude
        """
        force_mag = abs(risk_force)
        
        # --- 1. SAFETY Authority (Based on Field Force) ---
        # Low force -> Low Lambda (Human)
        # High force -> High Lambda (System)
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. PERFORMANCE Authority (Based on Gap Error) ---
        # 
        # Gap Error > 0: We are TOO FAR behind (need to catch up)
        # Gap Error < 0: We are TOO CLOSE (need to slow down)
        #
        # Both cases require system intervention when error is significant!
        
        lambda_performance = 0.1  # Default - human in control
        
        abs_gap_error = abs(gap_error)
        
        # THRESHOLD LOWERED from 10m to 2m for tighter tracking!
        if abs_gap_error > 2.0:
            # Progressive authority based on gap magnitude
            # At 2m error: lambda = 1.0
            # At 5m error: lambda = 1.0 + 3*0.3 = 1.9
            # At 10m error: lambda = 1.0 + 8*0.3 = 3.4
            # At 40m error: lambda = 1.0 + 38*0.3 = 12.4
            lambda_performance = 1.0 + (abs_gap_error - 2.0) * 0.3
            
            # Higher gain for POSITIVE gap error (catching up is more critical)
            # When we're behind, we need more aggressive system intervention
            if gap_error > 0:
                lambda_performance *= 1.5
            
        # Upper bound to prevent extreme values
        lambda_performance = min(lambda_performance, 50.0)

        # --- 3. Fusion: Take the MAX urgency ---
        # System takes control if there's safety risk OR performance issue
        target_lambda = max(lambda_safety, lambda_performance)
        
        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_performance = lambda_performance

        # --- 4. Smoothing ---
        # Prevent sudden jumps in authority
        lambda_k = (self.alpha_smoothing * target_lambda) + ((1 - self.alpha_smoothing) * self.prev_lambda)
        
        self.prev_lambda = lambda_k
        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        """Convert lambda to human/system weights."""
        total = 1.0 + lambda_k
        weight_human = 1.0 / total
        weight_system = lambda_k / total
        return weight_human, weight_system
    
    def reset(self):
        """Reset smoothing state."""
        self.prev_lambda = self.lambda_min


# Export
__all__ = ['LongitudinalAuthorityAllocator']
