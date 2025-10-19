#!/usr/bin/env python3
"""
File: longitudinal_authority_allocator.py
Description: Dynamic authority allocation based on Li et al. 2019 Table 2.

⭐ VERSION 2.2: FIXED - Separates pure lookup from runtime smoothing
"""

import numpy as np

class LongitudinalAuthorityAllocator:
    """
    Authority allocation based on Li et al. 2019 Table 2 (EXACT VALUES).
    
    ⭐ CRITICAL FIX: Smoothing only applied during runtime, not during lookup.
    
    Key behavior:
    - Force ≈ 0 → λ = 0.169 → Human dominant (85% human)
    - Force = +1000N → λ = 106.7 → System dominant (99% system) ✓
    - Negative forces → Lower authority (formation control)
    - Positive forces → Higher authority (safety control)
    """

    def __init__(self):
        # ⭐ EXACT Table 2 from Li et al. 2019 paper
        
        # Left side: Negative forces (attractive - too far)
        force_ranges_negative = np.array([-400, -350, -300, -250, -200, -150, -100, -50])
        ln_lambda_negative = np.array([-4.27, -4.27, -4.27, -4.25, -3.56, -2.82, -2.35, -2.01])
        
        # Right side: Positive forces (repulsive - too close)
        # ⭐ EXACT VALUES FROM THE PAPER!
        force_ranges_positive = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
        ln_lambda_positive = np.array([-1.78, -1.42, -0.83, -0.42, 0.00, 0.21, 0.63, 1.17, 1.61, 1.83, 2.21, 2.56, 3.34, 4.23, 4.67])
        
        # Combine into full table
        self.force_ranges = np.concatenate([force_ranges_negative, force_ranges_positive])
        ln_lambda_full = np.concatenate([ln_lambda_negative, ln_lambda_positive])
        self.lambda_values = np.exp(ln_lambda_full)
        
        # Runtime smoothing state (only used in real-time operation)
        self.prev_lambda = None
        self.smoothing_alpha = 0.8
        self.max_change_rate = 1.5
        
        print("🎯 Longitudinal Authority Allocator V2.2 (EXACT Table 2 - FIXED) initialized.")
        print(f"   Force range: [{self.force_ranges[0]}, {self.force_ranges[-1]}]N")
        print(f"   λ range: [{self.lambda_values.min():.3f}, {self.lambda_values.max():.1f}]")
        
        # Verify key points
        idx_0 = np.where(self.force_ranges == 0)[0][0]
        idx_1000 = np.where(self.force_ranges == 1000)[0][0]
        print(f"   At force=0: λ={self.lambda_values[idx_0]:.3f} (Human: {1/(1+self.lambda_values[idx_0])*100:.1f}%)")
        print(f"   At force=1000: λ={self.lambda_values[idx_1000]:.1f} (System: {self.lambda_values[idx_1000]/(1+self.lambda_values[idx_1000])*100:.1f}%)")

    def lookup_authority_ratio(self, risk_force: float) -> float:
        """
        ⭐ PURE lookup from table (no smoothing, no side effects).
        Use this for analysis, plotting, demonstrations.
        
        Args:
            risk_force: Risk force from safety field [N]
        
        Returns:
            Authority ratio λ(k) - pure table lookup
        """
        lambda_k = np.interp(
            risk_force,           
            self.force_ranges,    
            self.lambda_values    
        )
        return lambda_k

    def compute_authority_ratio(self, risk_force: float, use_smoothing: bool = True) -> float:
        """
        ⭐ Computes authority ratio with optional runtime smoothing.
        
        Args:
            risk_force: Risk force from safety field [N]
            use_smoothing: If True, applies smoothing and rate limiting
        
        Returns:
            Authority ratio λ(k)
        """
        # Pure lookup from table
        lambda_k = self.lookup_authority_ratio(risk_force)
        
        # Apply runtime smoothing if requested
        if use_smoothing and self.prev_lambda is not None:
            # Rate limiting
            max_increase = self.prev_lambda * self.max_change_rate
            min_decrease = self.prev_lambda / self.max_change_rate
            lambda_k_limited = np.clip(lambda_k, min_decrease, max_increase)
            
            # Adaptive smoothing
            alpha = self.smoothing_alpha
            if lambda_k > self.prev_lambda * 1.2:  # Emergency increase
                alpha = 0.7  # Less smoothing for quick response
            elif lambda_k < self.prev_lambda * 0.8:  # Quick decrease
                alpha = 0.9  # More smoothing for stability
            
            lambda_k = alpha * lambda_k_limited + (1 - alpha) * self.prev_lambda
        
        # Update state for next call (only if smoothing enabled)
        if use_smoothing:
            self.prev_lambda = lambda_k
        
        # Apply safety bounds
        lambda_k = np.clip(lambda_k, 0.01, 150.0)
        
        return lambda_k
    
    def reset_smoothing(self):
        """Reset smoothing state (useful for testing/demonstrations)."""
        self.prev_lambda = None
    
    def get_authority_weights(self, lambda_k: float) -> tuple:
        """
        Converts authority ratio to actual weights.
        
        Returns:
            (human_weight, system_weight) normalized to sum to 1
        """
        total = 1.0 + lambda_k
        human_weight = 1.0 / total
        system_weight = lambda_k / total
        return human_weight, system_weight


# ============================================================================
# EXAMPLES (FIXED - using pure lookup)
# ============================================================================

def demonstrate_full_table():
    """⭐ FIXED: Uses pure lookup (no smoothing side effects)."""
    print("\n" + "="*70)
    print("📊 AUTHORITY ALLOCATION V2.2 - EXACT Table 2 (PURE LOOKUP)")
    print("="*70)
    
    allocator = LongitudinalAuthorityAllocator()
    
    # Test key points from the paper (using PURE lookup)
    test_forces = [
        -400, -200, -100, -50, 
        0, 
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000
    ]
    
    print("\n" + "-"*70)
    print(f"{'Force (N)':>12} | {'ln(λ)':>8} | {'λ':>10} | {'Human%':>8} | {'System%':>8} | {'Status':>15}")
    print("-"*70)
    
    for force in test_forces:
        # ⭐ Use PURE lookup (no smoothing side effects)
        lambda_k = allocator.lookup_authority_ratio(force)
        human_w, system_w = allocator.get_authority_weights(lambda_k)
        ln_lambda = np.log(lambda_k)
        
        # Status indicator
        if system_w > 0.9:
            status = "🔴 SYSTEM"
        elif system_w > 0.7:
            status = "🟡 SYS LEAD"
        elif system_w >= 0.5:
            status = "🟢 BALANCED"
        elif system_w > 0.3:
            status = "🔵 HUM LEAD"
        else:
            status = "⚪ HUMAN"
        
        print(f"{force:+12.0f} | {ln_lambda:+8.2f} | {lambda_k:10.3f} | "
              f"{human_w*100:7.1f}% | {system_w*100:7.1f}% | {status:>15}")
        
        if force == 0:
            print("-"*70)
    
    print("-"*70)
    
    # ⭐ Verification against paper (using PURE lookup)
    print("\n✅ Verification against paper (PURE LOOKUP):")
    verifications = [
        (0, -1.78, 0.169),
        (50, -1.42, 0.242),
        (200, 0.00, 1.000),
        (500, 2.21, 9.116),
        (1000, 4.67, 106.6)
    ]
    
    print(f"{'Force':>8} | {'Paper ln(λ)':>12} | {'Paper λ':>10} | {'Code λ':>10} | {'Error':>10} | {'Match':>8}")
    print("-"*75)
    for force, paper_ln, paper_lambda in verifications:
        code_lambda = allocator.lookup_authority_ratio(force)
        error = abs(code_lambda - paper_lambda)
        match = "✅" if error < 0.1 else "⚠️" if error < 1.0 else "❌"
        print(f"{force:>8.0f} | {paper_ln:>12.2f} | {paper_lambda:>10.3f} | {code_lambda:>10.3f} | {error:>10.3f} | {match:>8}")
    
    print("\n" + "="*70)


def demonstrate_scenarios():
    """⭐ FIXED: Realistic scenarios with EXACT authority."""
    print("\n" + "="*70)
    print("🚗 REALISTIC DRIVING SCENARIOS (EXACT VALUES)")
    print("="*70)
    
    allocator = LongitudinalAuthorityAllocator()
    
    scenarios = [
        {
            'name': 'Perfect State',
            'force': 0,
            'paper_lambda': 0.169,
            'interpretation': 'Human drives, system monitors'
        },
        {
            'name': 'Slight Gap Issue',
            'force': 100,
            'paper_lambda': 0.436,
            'interpretation': 'System assists gently'
        },
        {
            'name': 'Moderate Risk',
            'force': 300,
            'paper_lambda': 1.878,
            'interpretation': 'System leads, human assists'
        },
        {
            'name': 'High Risk',
            'force': 500,
            'paper_lambda': 9.116,
            'interpretation': 'System in control'
        },
        {
            'name': 'Critical Emergency',
            'force': 800,
            'paper_lambda': 68.717,
            'interpretation': 'System emergency override'
        },
        {
            'name': 'Imminent Collision',
            'force': 1000,
            'paper_lambda': 106.6,
            'interpretation': 'Full system control'
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']} (Paper λ: {scenario['paper_lambda']:.3f})")
        
        force = scenario['force']
        lambda_k = allocator.lookup_authority_ratio(force)  # ⭐ Pure lookup
        human_w, system_w = allocator.get_authority_weights(lambda_k)
        
        error = abs(lambda_k - scenario['paper_lambda'])
        match = "✅" if error < 0.1 else "⚠️"
        
        print(f"   Force: {force:+.0f}N → λ = {lambda_k:.3f} {match}")
        print(f"   Authority: {human_w*100:.1f}% Human, {system_w*100:.1f}% System")
        print(f"   → {scenario['interpretation']}")
    
    print("\n" + "="*70)


def demonstrate_smoothing_effect():
    """⭐ NEW: Show the difference between pure lookup and smoothed."""
    print("\n" + "="*70)
    print("🔬 SMOOTHING EFFECT DEMONSTRATION")
    print("="*70)
    
    allocator = LongitudinalAuthorityAllocator()
    
    # Simulate sudden force changes
    force_sequence = [0, 500, 800, 200, 0]
    
    print("\nForce Sequence: " + " → ".join([f"{f}N" for f in force_sequence]))
    print("\n" + "-"*70)
    print(f"{'Step':>6} | {'Force':>8} | {'Pure λ':>10} | {'Smoothed λ':>12} | {'Difference':>12}")
    print("-"*70)
    
    allocator.reset_smoothing()  # Start fresh
    
    for step, force in enumerate(force_sequence, 1):
        pure_lambda = allocator.lookup_authority_ratio(force)
        smoothed_lambda = allocator.compute_authority_ratio(force, use_smoothing=True)
        diff = smoothed_lambda - pure_lambda
        
        print(f"{step:>6} | {force:>8.0f} | {pure_lambda:>10.3f} | {smoothed_lambda:>12.3f} | {diff:>+12.3f}")
    
    print("\n💡 Observations:")
    print("   • Pure lookup: Always gives EXACT table values")
    print("   • Smoothed: Gradually transitions (prevents sudden jumps)")
    print("   • Use pure lookup for: analysis, testing, demonstrations")
    print("   • Use smoothing for: real-time control (smoother driving)")
    print("="*70)


__all__ = ['LongitudinalAuthorityAllocator']

if __name__ == "__main__":
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    demonstrate_full_table()
    demonstrate_scenarios()
    demonstrate_smoothing_effect()
    
    print("\n✅ Authority allocator (V2.2 - PURE LOOKUP FIXED) demonstration completed!")
