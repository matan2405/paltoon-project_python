"""
Lateral Authority Allocator.

Research basis and adopted elements:
1) Li et al. (2019), authority-as-risk concept:
    - Safety authority increases with risk-force magnitude.
    - Used here as a sigmoid mapping from |risk_force| to lambda_safety.
2) Swain and Rath (2023), Eq. 15:
    - Replaces threshold/hysteresis lateral authority with continuous sigmoid weighting.
    - Implemented here via normalized lateral offset l_n and gamma1/gamma2 ratio.
3) Practical control policy in this codebase:
    - Fuses safety and lateral-offset authority by max(lambda_safety, lambda_lateral).
    - Applies adaptive smoothing (alpha_base/alpha_fast) to reduce authority chattering.

What this file contributes in code:
- Computes lambda(k) from safety and lateral offset channels.
- Stores per-channel lambdas for logging and post-analysis.
- Outputs stable authority trajectory for Nash shared steering.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUTHORITY_LAMBDA_MIN, AUTHORITY_LAMBDA_MAX,
    AUTHORITY_FORCE_MIDPOINT, AUTHORITY_K_STEEPNESS,
    AUTHORITY_ALPHA_BASE, AUTHORITY_ALPHA_FAST,
    AUTHORITY_ALPHA_FAST_THRESHOLD, AUTHORITY_ALPHA_BASE_THRESHOLD,
    AUTHORITY_SIGMOID_M1, AUTHORITY_SIGMOID_M2,
    LANE_WIDTH,
)


class LateralAuthorityAllocator:
    """
    Dynamic authority allocation with smooth sigmoid lateral-offset term.

    - Safety authority:   sigmoid on risk_force magnitude (unchanged)
    - Lateral-offset:     Swain & Rath sigmoid — continuous, no hysteresis
    - Fusion:             max(lambda_safety, lambda_lateral)
    - Adaptive smoothing: alpha_fast for large errors, alpha_base for small errors
    """

    def __init__(self):
        """Initialize safety and lateral-offset sigmoid mappings for authority ratio λ."""
        # Safety sigmoid parameters
        self.lambda_min = AUTHORITY_LAMBDA_MIN
        self.lambda_max = AUTHORITY_LAMBDA_MAX
        self.force_midpoint = AUTHORITY_FORCE_MIDPOINT
        self.k_steepness = AUTHORITY_K_STEEPNESS

        self.prev_lambda = self.lambda_min

        # Smoothing parameters
        self.alpha_base = AUTHORITY_ALPHA_BASE
        self.alpha_fast = AUTHORITY_ALPHA_FAST

        # Lateral-offset sigmoid parameters (Swain & Rath 2023, Eq. 15)
        self.sigmoid_m1 = AUTHORITY_SIGMOID_M1
        self.sigmoid_m2 = AUTHORITY_SIGMOID_M2
        self.l_o_max = LANE_WIDTH / 2.0   # max admissible lateral offset [m]

        # Store last computed values for logging
        self.last_lambda_safety = 0.0
        self.last_lambda_lateral = 0.0

        print(f"Authority Allocator (Safety + Swain&Rath Sigmoid) Initialized")
        print(f"   Safety sigmoid: λ [{self.lambda_min}, {self.lambda_max}], "
              f"F_mid={self.force_midpoint}N, k={self.k_steepness}")
        print(f"   Lateral sigmoid: m1={self.sigmoid_m1}, m2={self.sigmoid_m2}, "
              f"l_max={self.l_o_max:.2f}m  (Swain & Rath 2023, Eq. 15)")

    def compute_authority_ratio(self, risk_force: float,
                                 y_error: float = 0.0,
                                 heading_error: float = 0.0) -> float:
        """
        Compute authority ratio λ(k) using two smooth sigmoid sources.

        - Safety: sigmoid on |risk_force|
        - Lateral-offset: Swain & Rath sigmoid on |y_error| / l_o_max
        - Fusion: max(lambda_safety, lambda_lateral)
        - Adaptive smoothing based on error magnitude
        """
        force_mag = abs(risk_force)
        abs_y_error = abs(y_error)

        # --- 1. SAFETY Authority (sigmoid on risk force, unchanged) ---
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. LATERAL-OFFSET Authority (Swain & Rath 2023, Eq. 15) ---
        # l_n: normalized distance index (1 = lane centre, 0 = lane boundary)
        l_n = (self.l_o_max - abs_y_error) / self.l_o_max
        l_n = float(np.clip(l_n, 0.0, 1.0))

        # Driver/automation weights
        gamma1 = 1.0 / (1.0 + np.exp(self.sigmoid_m1 * (-l_n + self.sigmoid_m2)))
        gamma2 = 1.0 - gamma1

        # Convert to λ ratio (automation / human)
        lambda_lateral = gamma2 / max(gamma1, 1e-6)

        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_lateral = lambda_lateral

        # --- 3. Fusion: Take the MAX urgency ---
        target_lambda = max(lambda_safety, lambda_lateral)

        # --- 4. Adaptive Smoothing ---
        if abs_y_error > AUTHORITY_ALPHA_FAST_THRESHOLD:
            alpha = self.alpha_fast
        elif abs_y_error > AUTHORITY_ALPHA_BASE_THRESHOLD:
            blend = (abs_y_error - AUTHORITY_ALPHA_BASE_THRESHOLD) / (
                AUTHORITY_ALPHA_FAST_THRESHOLD - AUTHORITY_ALPHA_BASE_THRESHOLD)
            alpha = self.alpha_base + (self.alpha_fast - self.alpha_base) * blend
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


__all__ = ['LateralAuthorityAllocator']
