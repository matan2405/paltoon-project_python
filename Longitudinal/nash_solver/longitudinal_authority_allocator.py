#!/usr/bin/env python3
"""
File: longitudinal_authority_allocator.py
Description: Dynamic authority allocation based on Safety and Performance.

VERSION 5.0 - WITH HYSTERESIS
=============================
Key changes from V4:
1. HYSTERESIS: Different thresholds for entering/exiting gap-based authority
   - Enter system control when |gap_error| > 3.0m
   - Exit system control when |gap_error| < 1.0m
   - This prevents oscillations at the boundary
2. SMOOTHER transitions with adaptive smoothing
3. CONTINUOUS function instead of hard threshold
"""

import numpy as np


class LongitudinalAuthorityAllocator:
    """
    Dynamic authority allocation with HYSTERESIS to prevent oscillations.
    
    V5.2 MAXIMUM COMFORT TUNING:
    - Very low alpha values for extremely smooth authority transitions
    - Gradual handoffs prevent acceleration spikes
    """

    def __init__(self):
        # Sigmoid Parameters for SAFETY (Risk based)
        self.lambda_min = 0.1   # Human dominant when safe
        self.lambda_max = 100.0 # System dominant when dangerous
        self.force_midpoint = 400.0 
        self.k_steepness = 0.015

        self.prev_lambda = self.lambda_min
        
        # === SMOOTHING PARAMETERS - MAXIMUM COMFORT ===
        # Very low alpha = very slow, very smooth transitions
        # The trade-off: slower response but no jerky handoffs
        self.alpha_base = 0.05      # Base smoothing (was 0.08) - extremely smooth
        self.alpha_fast = 0.12      # Fast response (was 0.20) - still smooth
        
        # === HYSTERESIS STATE ===
        # Track if we're currently in "performance mode"
        self.in_performance_mode = False
        self.ENTER_THRESHOLD = 3.0   # Enter performance mode when |gap_error| > 3m
        self.EXIT_THRESHOLD = 1.0    # Exit performance mode when |gap_error| < 1m
        
        # Store last computed authority values for logging
        self.last_lambda_safety = 0.0
        self.last_lambda_performance = 0.0

        print(f"🛡️ Authority Allocator V5.2 (Maximum Comfort) Initialized")
        print(f"   Smoothing: alpha_base={self.alpha_base}, alpha_fast={self.alpha_fast}")
        print(f"   Hysteresis: enter={self.ENTER_THRESHOLD}m, exit={self.EXIT_THRESHOLD}m")

    def compute_authority_ratio(self, risk_force: float, gap_error: float = 0.0, velocity_error: float = 0.0) -> float:
        """
        Computes authority ratio lambda(k) with HYSTERESIS.
        
        V5 CHANGES:
        - Hysteresis: different thresholds for entering/exiting
        - Continuous smooth function instead of hard threshold
        - Adaptive smoothing based on error magnitude
        """
        force_mag = abs(risk_force)
        abs_gap_error = abs(gap_error)
        
        # --- 1. SAFETY Authority (Based on Field Force) ---
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. PERFORMANCE Authority with HYSTERESIS ---
        #
        # Hysteresis logic:
        # - If NOT in performance mode: enter when |error| > ENTER_THRESHOLD
        # - If IN performance mode: exit when |error| < EXIT_THRESHOLD
        #
        # This creates a "dead band" that prevents oscillations
        
        # Update hysteresis state
        if not self.in_performance_mode:
            if abs_gap_error > self.ENTER_THRESHOLD:
                self.in_performance_mode = True
        else:
            if abs_gap_error < self.EXIT_THRESHOLD:
                self.in_performance_mode = False
        
        # Compute performance authority
        lambda_performance = 0.1  # Default - human in control
        
        if self.in_performance_mode:
            # Use CONTINUOUS function instead of hard threshold
            # Soft sigmoid-like activation based on gap error
            # At 1m: lambda ≈ 1.0 (just entered)
            # At 3m: lambda ≈ 1.6
            # At 10m: lambda ≈ 3.7
            # At 40m: lambda ≈ 12.7
            
            # Normalized error (0 at EXIT_THRESHOLD, 1 at ENTER_THRESHOLD)
            normalized_error = (abs_gap_error - self.EXIT_THRESHOLD) / (self.ENTER_THRESHOLD - self.EXIT_THRESHOLD)
            normalized_error = max(0.0, normalized_error)  # Clamp to positive
            
            # Progressive authority
            lambda_performance = 1.0 + normalized_error * 0.3 + (abs_gap_error - self.EXIT_THRESHOLD) * 0.25
            
            # Higher gain for POSITIVE gap error (catching up is more critical)
            if gap_error > 0:
                lambda_performance *= 1.3
            
        # Upper bound
        lambda_performance = min(lambda_performance, 50.0)

        # --- 3. Fusion: Take the MAX urgency ---
        target_lambda = max(lambda_safety, lambda_performance)
        
        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_performance = lambda_performance

        # --- 4. ADAPTIVE Smoothing ---
        # Use faster smoothing when error is large (need quick response)
        # Use slower smoothing when error is small (need stability)
        if abs_gap_error > 5.0:
            alpha = self.alpha_fast
        elif abs_gap_error > 2.0:
            # Linear interpolation between fast and base
            alpha = self.alpha_base + (self.alpha_fast - self.alpha_base) * (abs_gap_error - 2.0) / 3.0
        else:
            alpha = self.alpha_base
        
        lambda_k = alpha * target_lambda + (1 - alpha) * self.prev_lambda
        
        self.prev_lambda = lambda_k
        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        """Convert lambda to human/system weights."""
        total = 1.0 + lambda_k
        weight_human = 1.0 / total
        weight_system = lambda_k / total
        return weight_human, weight_system
    
    def reset(self):
        """Reset smoothing state and hysteresis."""
        self.prev_lambda = self.lambda_min
        self.in_performance_mode = False


# Export
__all__ = ['LongitudinalAuthorityAllocator']
