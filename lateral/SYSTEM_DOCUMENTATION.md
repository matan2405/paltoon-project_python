# 📚 Lateral Control V2.7 - System Documentation

## 🏗️ מבנה המערכת

```
Lateral_V2/
├── config.py                    # הגדרות גלובליות
├── main_v2.py                   # נקודת כניסה (CLI)
├── main_with_menu.py            # נקודת כניסה (Interactive Menu)
│
├── vehicle/                     # מודל הרכב
│   ├── components.py            # מצב הרכב (State)
│   └── vehicle.py               # דינמיקה + State-Space
│
├── control/                     # בקרים
│   ├── human_driver.py          # מודל נהג אנושי (Stanley)
│   
│   └── platoon_control.py       # ניהול הפלטון
│
├── nash_solver/                 # ליבת האלגוריתם
│   ├── lateral_nash_solver.py   # פותר Nash Equilibrium
│   ├── lateral_safety_field.py  # שדה בטיחות + Phase Detection
│   ├── lateral_authority_allocator.py  # הקצאת סמכות (λ)
│   ├── system_reference_generator.py   # R1 - מסלול המערכת
│   └── human_reference_generator.py    # R2 - מסלול הנהג
│
├── simulation/
│   └── simulator.py             # מנוע הסימולציה
│
└── visualization/
    ├── plots.py                 # גרפים סטטיים
    └── animation.py             # אנימציה
```

---

## 📄 תיאור כל קובץ

### 1️⃣ config.py (142 שורות)
**תפקיד:** הגדרות גלובליות למערכת כולה

**תוכן עיקרי:**
```python
# Simulation
SIMULATION_DT = 0.01          # 100 Hz
LANE_WIDTH = 3.5              # meters

# Stanley Controller (Human Driver)
STANLEY_K_E_CAUTIOUS = 0.003
STANLEY_K_E_NORMAL = 0.005
STANLEY_K_E_AGGRESSIVE = 0.008

# Nash Solver
NASH_Q_Y = 10.0               # Weight on lateral error
NASH_Q_PSI = 5000.0           # Weight on heading error
NASH_R1 = 1000000.0           # System control cost
NASH_R2 = 1000000.0           # Human control cost (base)

# Driver Type - Behavioral parameters only (Nash weights are fixed)
NASH_DRIVER_PARAMS = {
    'cautious': {},
    'normal': {},
    'aggressive': {}
}
```

---

### 2️⃣ vehicle/vehicle.py (166 שורות)
**תפקיד:** מודל דינמי של הרכב (2-DOF Bicycle Model)

**משוואות המודל:**
$$\dot{y} = v_x \cdot \psi + v_x \cdot \delta$$
$$\dot{\psi} = \frac{v_x}{L} \cdot \delta$$

**מתודות עיקריות:**
- `step(delta, dt)` - צעד דינמי אחד
- `get_state_space_matrices(dt)` - מטריצות A, B, C לדיסקרטיזציה
- `get_state_vector()` - וקטור מצב [y, ψ, ẏ, ψ̇]

**השוואה:**

| היבט | Lateral V2 | Longitudinal | Lateral V1 |
|------|------------|--------------|------------|
| מודל | 2-DOF Bicycle | Double Integrator | 2-DOF Bicycle |
| מצבים | [y, ψ, ẏ, ψ̇] | [s, v, a] | [y, ψ, ẏ, ψ̇] |
| קלט | δ (steering) | F (force) | δ (steering) |
| דיסקרטיזציה | ZOH (expm) | ZOH (expm) | Euler (בעייתי!) |

---

### 3️⃣ vehicle/components.py (53 שורות)
**תפקיד:** הגדרת מבנה מצב הרכב

```python
@dataclass
class VehicleState:
    x: float = 0.0      # מיקום אורכי
    y: float = 0.0      # מיקום רוחבי
    psi: float = 0.0    # זווית heading
    y_dot: float = 0.0  # מהירות רוחבית
    psi_dot: float = 0.0  # מהירות זוויתית
```

---

### 4️⃣ control/human_driver.py (120 שורות)
**תפקיד:** מודל התנהגות הנהג האנושי (Stanley Controller)

**משוואת Stanley:**
$$\delta = -k_e \cdot y_{error} - k_\psi \cdot \psi_{error}$$

**תלות בסוג נהג:**

| סוג | k_e | k_psi | התנהגות |
|-----|-----|-------|---------|
| Cautious | 0.003 | 0.3 | עדין, איטי |
| Normal | 0.005 | 0.5 | מאוזן |
| Aggressive | 0.008 | 0.7 | חד, מהיר |

**השוואה:**

| היבט | Lateral V2 | Longitudinal | Lateral V1 |
|------|------------|--------------|------------|
| מודל נהג | Stanley | IDM/CTG | Stanley |
| פרמטרים | k_e, k_psi | v0, T, a, b | k_e, k_psi |
| תלות בסוג | ✅ כן | ✅ כן | ❌ לא |

---

### 5️⃣ control/human_input_interface.py (295 שורות)
**תפקיד:** ממשק אבסטרקטי לקלט הנהג

**מצבי עבודה:**
- `SIMULATION` - הנהג מדומה על ידי Stanley Controller
- `REAL_TIME` - קלט מגיע מנהג אמיתי (Unity)

```python
def get_human_steering(self, y_error, psi_error, velocity) -> float:
    if self.mode == DriverMode.SIMULATION:
        return self.simulated_driver.compute_stanley_steering(...)
    else:
        return self._real_time_steering  # From Unity
```

---

### 6️⃣ control/platoon_control.py (106 שורות)
**תפקיד:** ניהול שיירת הרכבים האוטונומיים

**מתודות:**
- `create_platoon(num_vehicles, leader_x, gap)` - יצירת פלטון
- `update(dt)` - עדכון מיקומי הרכבים
- `get_vehicles_as_obstacles()` - המרה לרשימת מכשולים

**השוואה:**

| היבט | Lateral V2 | Longitudinal |
|------|------------|--------------|
| מודל פלטון | פשוט (מהירות קבועה) | CACC מלא |
| String Stability | לא רלוונטי | ✅ מיושם |
| CTG Controller | ❌ | ✅ |

---

### 7️⃣ nash_solver/lateral_nash_solver.py (320 שורות)
**תפקיד:** ❤️ **ליבת האלגוריתם** - פתרון משוואות Nash Equilibrium

**מבוסס על Li et al. 2019, משוואות (5)-(10):**

**Cost Function - Player 1 (System):**
$$J_1 = \sum_{k=0}^{N_p} (z_k - R_1)^T Q (z_k - R_1) + \sum_{k=0}^{N_u} (u_1^T R_1 u_1 + u_2^T S_1 u_2)$$

**Cost Function - Player 2 (Human):**
$$J_2 = \sum_{k=0}^{N_p} (z_k - R_2)^T (\lambda Q) (z_k - R_2) + \sum_{k=0}^{N_u} (u_2^T R_2 u_2 + u_1^T S_2 u_1)$$

**שיטת פתרון:** Gauss-Seidel Iteration
```python
for iteration in range(max_iterations):
    # Fix u2, solve for u1
    u1_new = solve_player1_qp(u2_fixed)
    
    # Fix u1, solve for u2
    u2_new = solve_player2_qp(u1_new)
    
    if converged(u1_new, u2_new):
        break
```

**פלט משותף:**
$$\delta_{shared} = \frac{\lambda}{\lambda + 1} \cdot \delta_{system} + \frac{1}{\lambda + 1} \cdot \delta_{human}$$

**השוואה:**

| היבט | Lateral V2 | Longitudinal | Lateral V1 |
|------|------------|--------------|------------|
| שיטת פתרון | Gauss-Seidel | Gauss-Seidel | Gauss-Seidel |
| R1 ≠ R2 | ✅ כן (Driver Type) | ✅ כן | ❌ לא |
| מטריצות שונות | R1=R1, R2=f(driver) | R1≠R2 | R1=R2 |
| Prediction Horizon | Np=20 | Np=20 | Np=20 |
| Control Horizon | Nu=10 | Nu=10 | Nu=10 |

---

### 8️⃣ nash_solver/lateral_safety_field.py (327 שורות)
**תפקיד:** חישוב שדה בטיחות + Phase Detection

**מבוסס על Wang et al. 2015/2016:**

**Elliptic Safety Field:**
$$E(x,y) = \frac{(x-x_{obs})^2}{a^2} + \frac{(y-y_{obs})^2}{b^2}$$

$$F_{safety} = k \cdot \tanh\left(\frac{1}{E^2}\right) \cdot \frac{\partial E}{\partial y}$$

**Phase Detection (כמו במערכת האורכית):**
```
CRUISE → GAP_SEARCH → LANE_CHANGE → LANE_KEEPING → FOLLOWING
```

**תנאי מעבר ל-FOLLOWING:**
- |y_error| < 15% × lane_width
- |ψ| < 3°
- |ẏ| < 0.3 m/s
- יציבות למשך 5 שניות

**השוואה:**

| היבט | Lateral V2 | Longitudinal | Lateral V1 |
|------|------------|--------------|------------|
| שדה בטיחות | Elliptic 2D | Elliptic 1D | Elliptic 2D |
| פונקציית כוח | tanh (saturated) | tanh (saturated) | **Linear (בעייתי!)** |
| Lane Centering | ❌ **לא!** | לא רלוונטי | ✅ **כן (גרם לאוסילציות!)** |
| Phase Detection | ✅ 5 phases | ✅ 4 phases | ❌ לא היה |
| Soft Transition | ✅ | ✅ | ❌ |

---

### 9️⃣ nash_solver/lateral_authority_allocator.py (68 שורות)
**תפקיד:** חישוב יחס הסמכות λ

**מבוסס על Li et al. 2019:**
$$\lambda = \lambda_{base} \cdot f(F_{safety}) \cdot g(y_{error}, \psi_{error})$$

**התנהגות:**
- λ גבוה → יותר סמכות למערכת
- λ נמוך → יותר סמכות לנהג
- λ = 1 → חלוקה שווה (50%-50%)

---

### 🔟 nash_solver/system_reference_generator.py (194 שורות)
**תפקיד:** יצירת מסלול R1 (המערכת)

**פולינום מדרגה 5 (חלק מאוד):**
$$y(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

**תנאי שפה:**
- $y(0) = y_{start}$, $\dot{y}(0) = 0$, $\ddot{y}(0) = 0$
- $y(T) = y_{target}$, $\dot{y}(T) = 0$, $\ddot{y}(T) = 0$

**זמן מעבר:** T = 8 שניות (איטי ובטוח)

---

### 1️⃣1️⃣ nash_solver/human_reference_generator.py (222 שורות)
**תפקיד:** יצירת מסלול R2 (הנהג)

**פולינום מדרגה 3 (מהיר יותר):**
$$y(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$$

**זמן מעבר לפי סוג נהג:**

| סוג | T (שניות) | Max Heading |
|-----|-----------|-------------|
| Cautious | 7.0 | 10° |
| Normal | 6.0 | 12° |
| Aggressive | 5.0 | 15° |

**השוואה:**

| היבט | Lateral V2 | Longitudinal | Lateral V1 |
|------|------------|--------------|------------|
| R1 (System) | 5th order, T=8s | Position-based | ❌ לא היה |
| R2 (Human) | 3rd order, T=5-7s | Velocity-based | ❌ לא היה |
| R1 ≠ R2 | ✅ | ✅ | ❌ (אותו Reference) |

---

### 1️⃣2️⃣ simulation/simulator.py (530 שורות)
**תפקיד:** מנוע הסימולציה הראשי

**לולאת סימולציה:**
```python
for t in range(total_steps):
    # 1. Update platoon
    platoon.update(dt)
    
    # 2. Get human driver intent
    delta_human = human_driver.compute_steering()
    
    # 3. Compute safety field
    F_safety = safety_field.compute_force()
    
    # 4. Compute authority ratio
    lambda_k = authority_allocator.compute(F_safety, errors)
    
    # 5. Solve Nash equilibrium
    delta_sys, delta_hum = nash_solver.solve(state, R1, R2, lambda_k)
    
    # 6. Combine inputs
    delta_shared = (lambda_k * delta_sys + delta_hum) / (lambda_k + 1)
    
    # 7. Apply to vehicle
    vehicle.step(delta_shared, dt)
    
    # 8. Record data
    data.append(...)
```

---

### 1️⃣3️⃣ visualization/plots.py (262 שורות)
**תפקיד:** יצירת גרפים סטטיים

**גרפים שנוצרים:**
1. **Results Plot (3×3):** y, ψ, ẏ, steering, authority, safety field, error, ay, phase
2. **Nash Analysis:** Control inputs, λ ratio, authority split
3. **Trajectory:** מסלול מעוף הציפור

---

### 1️⃣4️⃣ visualization/animation.py (364 שורות)
**תפקיד:** אנימציה אינטראקטיבית

**תכונות:**
- Bird's eye view עם רכבים
- Time series plots בזמן אמת
- צביעה לפי Phase
- שמירה כ-GIF

---

### 1️⃣5️⃣ main_with_menu.py (534 שורות)
**תפקיד:** ממשק אינטראקטיבי

**תפריט:**
```
1. Scenario 1 - Join Before Platoon
2. Scenario 2 - Join Middle of Platoon
3. Scenario 3 - Join After Platoon
4. Run all scenarios
5. Run single scenario with driver selection
6. Exit
```

---

**תפקיד:** בקר לעבודה עם Unity בזמן אמת

**מתודות:**
- `process_unity_state(state_dict)` - קבלת מצב מ-Unity
- `get_control_output()` - שליחת פקודת בקרה

---

**תפקיד:** תקשורת TCP עם Unity

**פרוטוקול:**
- Port: 5000
- Format: JSON
- Bidirectional: State ← Unity, Control → Unity

---

## 📊 השוואה מסכמת

### Lateral V2 vs Lateral V1

| היבט | V1 (כושל) | V2 (מוצלח) |
|------|-----------|------------|
| **תוצאה** | אוסילציות ±120m | התכנסות מושלמת |
| **Lane Centering** | ✅ (גרם לקונפליקט!) | ❌ (הוסר!) |
| **פונקציית כוח** | Linear | tanh (saturated) |
| **Phase Detection** | ❌ | ✅ 5 phases |
| **R1 ≠ R2** | ❌ | ✅ |
| **Driver Type in Nash** | ❌ | ✅ |
| **דיסקרטיזציה** | Euler | ZOH (expm) |

### Lateral V2 vs Longitudinal

| היבט | Lateral V2 | Longitudinal |
|------|------------|--------------|
| **מודל רכב** | 2-DOF Bicycle | Double Integrator |
| **מצבים** | [y, ψ, ẏ, ψ̇] | [s, v, a] |
| **קלט** | δ (steering) | F (force) |
| **מודל נהג** | Stanley | IDM |
| **Phase Detection** | ✅ (זהה) | ✅ |
| **Nash Solver** | ✅ (זהה) | ✅ |
| **Safety Field** | 2D Elliptic | 1D Elliptic |
| **String Stability** | לא רלוונטי | ✅ CTG |

---

## 🎯 הלקחים העיקריים

### 1. **הסרת Lane Centering**
הכוח של Lane Centering **התנגש** עם ה-Stanley Controller → אוסילציות
**פתרון:** Nash solver מטפל במעקב, Safety Field רק מרחיק ממכשולים

### 2. **Phase Detection**
מאפשר התנהגות שונה בכל שלב:
- LANE_CHANGE: כוחות בטיחות מלאים
- FOLLOWING: הפחתת כוחות (soft transition)

### 3. **R1 ≠ R2**
לפי Li et al. 2019 - מסלול המערכת שונה ממסלול הנהג:
- System: פולינום מדרגה 5, T=8s (חלק ובטוח)
- Human: פולינום מדרגה 3, T=5-7s (מהיר יותר)

### 4. **Driver Type Dependency**
פרמטרי Nash משתנים לפי סוג הנהג:
- R2, S2, Q_y מותאמים לאישיות

### 5. **Consistent Architecture**
אותה ארכיטקטורה כמו Longitudinal:
- Gauss-Seidel Nash solver
- Phase detection with hysteresis
- Soft transitions
- ZOH discretization
