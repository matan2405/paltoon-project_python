#!/usr/bin/env python3
"""
File: longitudinal_authority_allocator.py
Description: Contains the dynamic authority allocation logic for longitudinal control.
"""

import numpy as np
class LongitudinalAuthorityAllocator:
    """
    Enhanced authority allocation for longitudinal control based on Li et al. Table 2.
    Determines the dynamic authority allocation ratio (lambda_k) based on longitudinal driving risk.
    """

    def __init__(self):
        # Li et al. Table 2 - Force ranges and their corresponding ln(lambda) values.
        # Original paper: Force=0 -> ln(lambda)=-1.78 -> lambda=0.169 (Human dominant!)
        self.force_ranges = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
        
        # Convert from ln(lambda) to lambda values from Table 2
        ln_lambda = np.array([-1.78, -1.42, -0.83, -0.42, 0.00, 0.21, 0.63, 1.17, 1.61, 1.83, 2.21, 2.56, 3.34, 4.23, 4.67])
        self.lambda_values = np.exp(ln_lambda)

        # Enhanced smoothing parameters for stability
        self.prev_lambda = None  # Start with human dominant (as per paper)
        self.smoothing_alpha = 0.8  # Increased from 0.7 to 0.8 for more smoothing
        self.max_change_rate = 1.5  # Reduced from 2.0 to 1.5 for smoother transitions

        print("Enhanced Longitudinal Authority Allocator initialized.")

    def compute_authority_ratio(self, risk_force: float) -> float:
        """
        Computes the authority ratio lambda(k) based on the scalar longitudinal risk force.

        Args:
            risk_force (float): The risk value from the LongitudinalSafetyField.

        Returns:
            float: The calculated authority ratio lambda(k).
        """
        # Improved: Enhanced lookup with interpolation
        force_mag = abs(risk_force)  # Use absolute value for scalar force
        lambda_k = np.interp(force_mag, self.force_ranges, self.lambda_values)

        # Enhanced: Adaptive response for high longitudinal risk scenarios
        if force_mag > 200.0:  # High longitudinal risk (hard braking/acceleration scenarios)
            lambda_k *= 1.2  # Reduced multiplier from 1.5 to 1.2 for stability

        # Rate limiting with increased responsiveness
        if self.prev_lambda is None:
            max_increase = lambda_k * self.max_change_rate
            min_decrease = lambda_k / self.max_change_rate
        else:
            max_increase = self.prev_lambda * self.max_change_rate
            min_decrease = self.prev_lambda / self.max_change_rate
        # lambda_k = np.clip(lambda_k, min_decrease, max_increase)

        # Enhanced: Adaptive smoothing based on risk level changes
        alpha = self.smoothing_alpha
        if self.prev_lambda is not None:
            if lambda_k > self.prev_lambda * 1.2:  # Emergency increase
                alpha = 0.8  # Increased from 0.7 to 0.8 for smoother emergency response
            elif lambda_k < self.prev_lambda * 0.8:  # Quick decrease
                alpha = 0.9

            # Apply smoothing with adaptive alpha
            lambda_k = alpha * lambda_k + (1 - alpha) * self.prev_lambda

        # Apply bounds - Reduced upper bound for stability
        lambda_k = np.clip(lambda_k, 0.05, 150.0)

        self.prev_lambda = lambda_k
        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        """
        Converts authority ratio to actual weights.
        
        Args:
            lambda_k (float): The authority ratio
            
        Returns:
            tuple: (human_weight, autonomous_weight) normalized to sum to 1
        """
        total = 1 + lambda_k
        human_weight = 1.0 / total
        autonomous_weight = lambda_k / total
        return human_weight, autonomous_weight


# ================================
# Detailed Implementation Example
# ================================

def longitudinal_control_example():
    """
    Detailed example of using LongitudinalAuthorityAllocator
    in autonomous vehicle longitudinal control system.
    """
    
    print("🚗 Example: Autonomous Longitudinal Control System with Dynamic Authority Allocation")
    print("=" * 80)
    
    # Create authority allocator
    allocator = LongitudinalAuthorityAllocator()
    
    # Example scenarios
    scenarios = [
        {"name": "Normal Driving", "risk_force": 25.0, "description": "Constant speed, safe distance"},
        {"name": "Mild Approach", "risk_force": 120.0, "description": "Approaching leading vehicle"},
        {"name": "Emergency Braking", "risk_force": 350.0, "description": "Leading vehicle braked hard"},
        {"name": "Critical Risk", "risk_force": 600.0, "description": "Emergency situation - collision risk"},
        {"name": "Return to Calm", "risk_force": 80.0, "description": "Situation stabilizing"}
    ]
    
    print("\n🎯 Running scenario simulation:")
    print("-" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"📊 Risk Force: {scenario['risk_force']:.1f}")
        print(f"📝 Description: {scenario['description']}")
        
        # Calculate authority ratio
        lambda_k = allocator.compute_authority_ratio(scenario['risk_force'])
        
        # Get weights
        human_weight, auto_weight = allocator.get_authority_weights(lambda_k)
        
        # Print results
        print(f"⚖️  Authority Ratio (lambda): {lambda_k:.3f}")
        print(f"👤 Human Weight: {human_weight:.3f} ({human_weight*100:.1f}%)")
        print(f"🤖 Autonomous Weight: {auto_weight:.3f} ({auto_weight*100:.1f}%)")
        
        # Interpret results
        if auto_weight > 0.7:
            authority_status = "🔴 AUTONOMOUS DOMINANT - System in control"
        elif auto_weight > 0.5:
            authority_status = "🟡 AUTONOMOUS LEAD - System leading"
        elif human_weight > 0.7:
            authority_status = "🔵 HUMAN DOMINANT - Driver in control"
        else:
            authority_status = "🟢 BALANCED - Cooperative control"
            
        print(f"🚦 Status: {authority_status}")
        
        # Example of practical use - combined control command calculation
        human_command = np.random.uniform(-2.0, 1.0)  # Example driver command
        auto_command = np.random.uniform(-3.0, 2.0)   # Example system command
        
        final_command = human_weight * human_command + auto_weight * auto_command
        
        print(f"🎛️  Driver Command: {human_command:.2f} m/s^2")
        print(f"🎛️  System Command: {auto_command:.2f} m/s^2")
        print(f"➡️  Final Command: {final_command:.2f} m/s^2")


def integration_example():
    """
    Example of integration with Nash Game Solver system
    """
    
    print("\n" + "=" * 80)
    print("🔧 Example: Integration with Nash Game Solver")
    print("=" * 80)
    
    allocator = LongitudinalAuthorityAllocator()
    
    # Simulate control loop
    print("\n🔄 Control Loop (5 time steps):")
    print("-" * 40)
    
    for step in range(5):
        # Simulate sensor data reception
        simulated_risk = 50 + step * 100 + np.random.normal(0, 20)
        simulated_risk = max(0, simulated_risk)  # Ensure positive
        
        print(f"\nStep {step + 1}:")
        print(f"📡 Calculated Risk Force: {simulated_risk:.1f}")
        
        # Calculate authority ratio
        lambda_k = allocator.compute_authority_ratio(simulated_risk)
        human_w, auto_w = allocator.get_authority_weights(lambda_k)
        
        print(f"⚖️  Authority Ratio: {lambda_k:.3f}")
        print(f"👥 Distribution: Human {human_w:.2f} | Autonomous {auto_w:.2f}")
        
        # Example strategy determination
        if auto_w > 0.6:
            strategy = "🔴 System taking control"
        elif human_w > 0.6:
            strategy = "🔵 Driver in control"
        else:
            strategy = "🟢 Cooperative control"
            
        print(f"🎯 Strategy: {strategy}")

__all__ = ['LongitudinalAuthorityAllocator']
if __name__ == "__main__":
    # Run examples
    longitudinal_control_example()
    integration_example()
    
    print("\n" + "🎉 Example completed successfully! 🎉")
    print("💡 You can use this class in your longitudinal control system.")
