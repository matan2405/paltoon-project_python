# #!/usr/bin/env python3
# """
# File: longitudinal_authority_allocator.py
# Description: Dynamic authority allocation based on Li et al. 2019 Table 2.
# Updated: Simplified to prevent oscillations.
# """

# import numpy as np

# class LongitudinalAuthorityAllocator:
#     def __init__(self):
#         # Standard Table setup
#         force_ranges_negative = np.array([-400, -350, -300, -250, -200, -150, -100, -50])
#         ln_lambda_negative = np.array([-4.27, -4.27, -4.27, -4.25, -3.56, -2.82, -2.35, -2.01])
#         force_ranges_positive = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000])
#         ln_lambda_positive = np.array([-1.78, -1.42, -0.83, -0.42, 0.00, 0.21, 0.63, 1.17, 1.61, 1.83, 2.21, 2.56, 3.34, 4.23, 4.67])
        
#         self.force_ranges = np.concatenate([force_ranges_negative, force_ranges_positive])
#         ln_lambda_full = np.concatenate([ln_lambda_negative, ln_lambda_positive])
#         self.lambda_values = np.exp(ln_lambda_full)
        
#         self.prev_lambda = None
#         self.smoothing_alpha = 0.8
#         self.max_change_rate = 1.5
#         print("🎯 Longitudinal Authority Allocator Initialized.")

#     def lookup_authority_ratio(self, risk_force: float) -> float:
#         return np.interp(risk_force, self.force_ranges, self.lambda_values)

#     def compute_authority_ratio(self, risk_force: float, gap_error: float = 0.0, velocity_error: float = 0.0, use_smoothing: bool = True) -> float:
#         """
#         Computes authority ratio based on Risk and Gap Performance.
#         Removed Velocity Error override to prevent oscillations.
#         """
#         # 1. Risk Based
#         lambda_risk = self.lookup_authority_ratio(risk_force)
        
#         # 2. Performance Based (Gap Closing)
#         lambda_performance = 0.0
        
#         error_tolerance = 1.0
        
#         if gap_error > error_tolerance:
#             # Moderate slope
#             slope = 0.5
#             lambda_performance = (gap_error - error_tolerance) * slope
            
#         # Fusion
#         lambda_k = max(lambda_risk, lambda_performance)

#         # Smoothing
#         if use_smoothing and self.prev_lambda is not None:
#             max_increase = self.prev_lambda * self.max_change_rate
#             min_decrease = self.prev_lambda / self.max_change_rate
#             lambda_k = np.clip(lambda_k, min_decrease, max_increase)
            
#             alpha = self.smoothing_alpha
#             if lambda_k > self.prev_lambda * 1.2: alpha = 0.7
#             elif lambda_k < self.prev_lambda * 0.8: alpha = 0.9 
            
#             lambda_k = alpha * lambda_k + (1 - alpha) * self.prev_lambda
        
#         self.prev_lambda = lambda_k
#         return np.clip(lambda_k, 0.01, 150.0)
        
#     def reset_smoothing(self):
#         self.prev_lambda = None
        
#     def get_authority_weights(self, lambda_k: float) -> tuple:
#         total = 1.0 + lambda_k
#         return 1.0 / total, lambda_k / total

# __all__ = ['LongitudinalAuthorityAllocator']

#!/usr/bin/env python3
import numpy as np

class LongitudinalAuthorityAllocator:
    """
    Dynamic authority allocation based on a continuous Sigmoid function.
    Updated: Includes PERFORMANCE logic to force system control when trailing behind.
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

        print(f"🛡️ Authority Allocator V3 (Risk + Gap Performance) Initialized")

    def compute_authority_ratio(self, risk_force: float, gap_error: float = 0.0, velocity_error: float = 0.0) -> float:
        """
        Computes authority ratio lambda(k).
        Logic: Max(Safety_Need, Performance_Need)
        """
        force_mag = abs(risk_force)
        
        # --- 1. SAFETY Authority (Based on Field Force) ---
        # Low force -> Low Lambda (Human)
        # High force -> High Lambda (System)
        sigmoid_factor = 1.0 / (1.0 + np.exp(-self.k_steepness * (force_mag - self.force_midpoint)))
        lambda_safety = self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid_factor

        # --- 2. PERFORMANCE Authority (Based on Gap) ---
        # אם אנחנו רחוקים מדי (Gap Error חיובי וגדול), המערכת חייבת לקחת פיקוד כדי להדביק את הקצב.
        # הנהג האנושי "נרדם" או איטי מדי.
        lambda_performance = 0.1 # Default
        
        # אם הפער גדול מ-10 מטר מהרצוי, נתחיל להעביר סמכות למערכת
        if gap_error > 10.0:
            # פונקציה ליניארית: על כל מטר נוסף של פער, נוסיף סמכות למערכת
            # בפער של 100 מטר, נקבל lambda=~45 (שליטה כמעט מלאה של המערכת)
            lambda_performance = 1.0 + (gap_error - 10.0) * 0.5
            
        # חוסם עליון לביצועים (לא חייבים 100, מספיק 50 כדי לשלוט)
        lambda_performance = min(lambda_performance, 50.0)

        # --- 3. Fusion: Take the MAX urgency ---
        # המערכת לוקחת שליטה אם יש סכנה או אם הביצועים גרועים
        target_lambda = max(lambda_safety, lambda_performance)
        
        # Store for logging
        self.last_lambda_safety = lambda_safety
        self.last_lambda_performance = lambda_performance

        # --- 4. Smoothing ---
        # מניעת קפיצות פתאומיות
        lambda_k = (self.alpha_smoothing * target_lambda) + ((1 - self.alpha_smoothing) * self.prev_lambda)
        
        self.prev_lambda = lambda_k
        return lambda_k

    def get_authority_weights(self, lambda_k: float) -> tuple:
        total = 1.0 + lambda_k
        weight_human = 1.0 / total
        weight_system = lambda_k / total
        return weight_human, weight_system