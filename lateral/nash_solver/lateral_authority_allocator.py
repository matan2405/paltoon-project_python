"""
Lateral Authority Allocator.
VERSION 3.0 - Aligned with Longitudinal V5.2

Key changes from V2.2:
1. HYSTERESIS on y_error (mirrors gap_error logic in longitudinal):
   - Enter performance mode when |y_error| > 1.0m
   - Exit performance mode when |y_error| < 0.3m
2. TWO sources fused by max(): SAFETY (sigmoid on force) + PERFORMANCE (hysteresis)
3. ADAPTIVE smoothing: faster when error is large, slower when small
4. Removed rate limiting (replaced by adaptive alpha - cleaner)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUTHORITY_LAMBDA_MIN, AUTHORITY_LAMBDA_MAX,
    AUTHORITY_FORCE_MIDPOINT, AUTHORITY_K_STEEPNESS,
    AUTHORITY_ALPHA_BASE, AUTHORITY_ALPHA_FAST,
    AUTHORITY_ENTER_THRESHOLD, AUTHORITY_EXIT_THRESHOLD,
    AUTHORITY_LAMBDA_PERFORMANCE_MAX,
)


class LateralAuthorityAllocator:
    """
    Dynamic authority allocation with HYSTERESIS to prevent oscillations.

    V3.0 - Aligned with Longitudinal V5.2:
    - Safety authority: sigmoid on risk_force magnitude
    - Performance authority: hysteresis on |y_error| (like gap_error in longitudinal)
    - Fusion: max(lambda_safety, lambda_performance)
    - Adaptive smoothing: alpha_fast for large errors, alpha_base for small errors
    """

    def __init__(self):
        # Sigmoid Parameters for SAFETY (risk-based)
        self.lambda_min = AUTHORITY_LAMBDA_MIN
        self.lambda_max = AUTHORITY_LAMBDA_MAX
        self.force_midpoint = AUTHORITY_FORCE_MIDPOINT
        self.k_steepness = AUTHORITY_K_STEEPNESS

        self.prev_lambda = self.lambda_min

        # Smoothing parameters
        self.alpha_base = AUTHORITY_ALPHA_BASE
        self.alpha_fast = AUTHORITY_ALPHA_FAST

        # Hysteresis state
        self.in_performance_mode = False
        self.ENTER_THRESHOLD = AUTHORITY_ENTER_THRESHOLD
        self.EXIT_THRESHOLD = AUTHORITY_EXIT_THRESHOLD
        self.lambda_performance_max = AUTHORITY_LAMBDA_PERFORMANCE_MAX

        # Store last computed values for logging
        self.last_lambda_safety = 0.0
        self.last_lambda_performance = 0.0

        print(f"🎯 Authority Allocator V3.0 (Hysteresis + Adaptive) Initialized")
        print(f"   λ range: {self.lambda_min}-{self.lambda_max}")
        print(f"   Smoothing: alpha_base={self.alpha_base}, alpha_fast={self.alpha_fast}")
        print(f"   Hysteresis: enter={self.ENTER_THRESHOLD}m, exit={self.EXIT_THRESHOLD}m")

    def compute_authority_ratio(self, risk_force: float,
                                 y_error: float = 0.0,
                                 heading_error: float = 0.0) -> float:
        """
        Compute authority ratio λ(k) with HYSTERESIS.

        V3.0:
        - Two sources: SAFETY (sigmoid on force) + PERFORMANCE (hysteresis on y_error)
        - Fusion: max(lambda_safety, lambda_performance)
        - Adaptive smoothing based on error magnitude
        """
        force_mag = abs(risk_force)
        abs_y_error = abs(y_error)

        # --- 1. SAFETY Authority (sigmoid on risk force) ---
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. PERFORMANCE Authority with HYSTERESIS on y_error ---
        if not self.in_performance_mode:
            if abs_y_error > self.ENTER_THRESHOLD:
                self.in_performance_mode = True
        else:
            if abs_y_error < self.EXIT_THRESHOLD:
                self.in_performance_mode = False

        lambda_performance = 0.1  # Default - human in control

        if self.in_performance_mode:
            # Normalized error (0 at EXIT_THRESHOLD, 1 at ENTER_THRESHOLD)
            normalized_error = (abs_y_error - self.EXIT_THRESHOLD) / (self.ENTER_THRESHOLD - self.EXIT_THRESHOLD)
            normalized_error = max(0.0, normalized_error)

            # Progressive authority
            lambda_performance = 1.0 + normalized_error * 0.3 + (abs_y_error - self.EXIT_THRESHOLD) * 0.25

            # Higher gain when y_error is positive (ego above target - more critical to correct)
            if y_error > 0:
                lambda_performance *= 1.3

        lambda_performance = min(lambda_performance, self.lambda_performance_max)

        # --- 3. Fusion: Take the MAX urgency ---
        target_lambda = max(lambda_safety, lambda_performance)

        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_performance = lambda_performance

        # --- 4. Adaptive Smoothing ---
        if abs_y_error > 1.5:
            alpha = self.alpha_fast
        elif abs_y_error > 0.5:
            alpha = self.alpha_base + (self.alpha_fast - self.alpha_base) * (abs_y_error - 0.5) / 1.0
        else:
            alpha = self.alpha_base

        lambda_k = alpha * target_lambda + (1 - alpha) * self.prev_lambda
        self.prev_lambda = lambda_k

        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        """Convert λ to human/system weights."""
        total = 1.0 + lambda_k
        return 1.0 / total, lambda_k / total

    def reset(self):
        """Reset state for new simulation."""
        self.prev_lambda = self.lambda_min
        self.in_performance_mode = False


__all__ = ['LateralAuthorityAllocator']
