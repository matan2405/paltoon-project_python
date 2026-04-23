#!/usr/bin/env python3
"""
File: longitudinal_authority_allocator.py
Description: Dynamic authority allocation based on Safety and Performance.

Research basis and implemented policy:
1. Safety-based authority from risk-force sigmoid mapping.
2. Swain and Rath (2023, Eq. 15) style smooth sigmoid for gap/velocity error.
3. max() fusion and adaptive smoothing for stable shared-control handoff.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUTHORITY_LAMBDA_MIN, AUTHORITY_LAMBDA_MAX,
    AUTHORITY_FORCE_MIDPOINT, AUTHORITY_K_STEEPNESS,
    AUTHORITY_ALPHA_BASE, AUTHORITY_ALPHA_FAST,
    AUTHORITY_SIGMOID_M1, AUTHORITY_SIGMOID_M2,
    AUTHORITY_GAP_ERROR_MAX, AUTHORITY_VEL_ERROR_MAX,
)


class LongitudinalAuthorityAllocator:
    """
    Dynamic authority allocation with smooth sigmoid gap-error term.

    Swain & Rath 2023, Eq. 15 mapping (adapted for gap error):
    - Safety authority:  sigmoid on risk_force magnitude (unchanged)
    - Gap-error:         Swain & Rath sigmoid — continuous, no hysteresis
    - Fusion:            max(lambda_safety, lambda_gap)
    - Adaptive smoothing: alpha_fast for large errors, alpha_base for small errors
    """

    def __init__(self):
        """Initialize sigmoid authority mapping parameters and smoothing state."""
        # Safety sigmoid parameters
        self.lambda_min = AUTHORITY_LAMBDA_MIN
        self.lambda_max = AUTHORITY_LAMBDA_MAX
        self.force_midpoint = AUTHORITY_FORCE_MIDPOINT
        self.k_steepness = AUTHORITY_K_STEEPNESS

        self.prev_lambda = self.lambda_min

        # Smoothing parameters
        self.alpha_base = AUTHORITY_ALPHA_BASE
        self.alpha_fast = AUTHORITY_ALPHA_FAST

        # Gap-error + velocity-error sigmoid parameters (Swain & Rath 2023, Eq. 15)
        self.sigmoid_m1 = AUTHORITY_SIGMOID_M1
        self.sigmoid_m2 = AUTHORITY_SIGMOID_M2
        self.gap_error_max = AUTHORITY_GAP_ERROR_MAX  # max admissible gap error [m]
        self.vel_error_max = AUTHORITY_VEL_ERROR_MAX  # max admissible velocity error [m/s]

        # Store last computed values for logging
        self.last_lambda_safety = 0.0
        self.last_lambda_gap = 0.0

        print(f"Authority Allocator (Safety + Swain&Rath Sigmoid) Initialized")
        print(f"   Safety sigmoid: λ [{self.lambda_min}, {self.lambda_max}], "
              f"F_mid={self.force_midpoint}N, k={self.k_steepness}")
        print(f"   Gap+Vel sigmoid: m1={self.sigmoid_m1}, m2={self.sigmoid_m2}, "
              f"gap_max={self.gap_error_max:.2f}m, vel_max={self.vel_error_max:.2f}m/s  (Swain & Rath 2023, Eq. 15)")

    def compute_authority_ratio(self, risk_force: float,
                                 gap_error: float = 0.0,
                                 velocity_error: float = 0.0) -> float:
        """
        Compute authority ratio λ(k) using two smooth sigmoid sources.

        Notes:
        - Safety: sigmoid on |risk_force|
        - Gap+Velocity: Swain & Rath sigmoid on min(l_n_gap, l_n_vel) — most demanding error wins
        - Fusion: max(lambda_safety, lambda_gap)
        - Adaptive smoothing based on max(|gap_error|, |vel_error|)
        """
        force_mag = abs(risk_force)
        abs_gap_error = abs(gap_error)

        # --- 1. SAFETY Authority (sigmoid on risk force, unchanged) ---
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. GAP+VELOCITY Authority (Swain & Rath 2023, Eq. 15 — combined error index) ---
        # l_n: normalized index (1 = no error, 0 = max error / full system authority)
        # Both gap error and velocity error reduce l_n; the most demanding one wins.
        abs_vel_error = abs(velocity_error)
        l_n_gap = (self.gap_error_max - abs_gap_error) / self.gap_error_max
        l_n_vel = (self.vel_error_max - abs_vel_error) / self.vel_error_max
        l_n = float(np.clip(min(l_n_gap, l_n_vel), 0.0, 1.0))

        # Driver/automation weights
        gamma1 = 1.0 / (1.0 + np.exp(self.sigmoid_m1 * (-l_n + self.sigmoid_m2)))
        gamma2 = 1.0 - gamma1

        # Convert to λ ratio (automation / human)
        lambda_gap = gamma2 / max(gamma1, 1e-6)

        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_gap = lambda_gap

        # --- 3. Fusion: Take the MAX urgency ---
        target_lambda = max(lambda_safety, lambda_gap)

        # --- 4. Adaptive Smoothing ---
        # Use whichever error is larger (gap or velocity) to set smoothing speed
        combined_error = max(abs_gap_error, abs_vel_error)
        if combined_error > 5.0:
            alpha = self.alpha_fast
        elif combined_error > 2.0:
            alpha = self.alpha_base + (self.alpha_fast - self.alpha_base) * (combined_error - 2.0) / 3.0
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


# Export
__all__ = ['LongitudinalAuthorityAllocator']
