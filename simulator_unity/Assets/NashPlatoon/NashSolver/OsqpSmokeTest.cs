using System;
using UnityEngine;
using OsqpSharp;

/// <summary>
/// Smoke-test for OsqpSharp (pure C# ADMM QP solver).
/// Attach to any GameObject — runs on Start() and via the right-click context menu.
///
/// QP:  min 0.5·x²   s.t.  1 ≤ x ≤ 10   →  solution x = 1.0
/// OSQP form:  min 0.5·x'Px + q'x,  P=[1], q=[0],  A=[1],  l=[1],  u=[10]
/// </summary>
[DefaultExecutionOrder(-200)]
public class OsqpSmokeTest : MonoBehaviour
{
    void Start() => RunTest();

    [ContextMenu("Run OSQP Smoke Test")]
    public void RunTestMenu() => RunTest();

    public static bool RunTest()
    {
        try
        {
            const int n = 1, m = 1;

            // P = [[1]]  upper-triangular CSC
            double[] Pdata = { 1.0 };
            int[]    Pi    = { 0 };
            int[]    Pp    = { 0, 1 };

            double[] q = { 0.0 };

            // A = [[1]]  CSC
            double[] Adata = { 1.0 };
            int[]    Ai    = { 0 };
            int[]    Ap    = { 0, 1 };

            double[] lo = { 1.0 };
            double[] hi = { 10.0 };

            var settings = new OsqpSettings
            {
                Verbose           = false,
                WarmStarting      = false,
                MaxIter           = 500,
                AbsoluteTolerance = 1e-6,
                RelativeTolerance = 1e-6,
            };

            var solver = new OsqpSolver(settings);
            solver.Setup(n, m, Pdata, Pi, Pp, q, Adata, Ai, Ap, lo, hi);
            solver.Solve();

            double x  = solver.Solution.X[0];
            bool   ok = Math.Abs(x - 1.0) < 1e-3;

            if (ok)
                Debug.Log($"[OsqpSmokeTest] PASS — x = {x:F6}  (expected 1.0), " +
                          $"status = {solver.Solution.Status}, " +
                          $"iters = {solver.Solution.Iterations}");
            else
                Debug.LogError($"[OsqpSmokeTest] FAIL — x = {x:F6}  (expected 1.0), " +
                               $"status = {solver.Solution.Status}");

            // ── Second test: warm-start update ────────────────────────────
            // OSQP: min 0.5·x² + q·x  (P=[1] unchanged)
            // Set q=-3: min 0.5x² - 3x → d/dx = x-3 = 0 → x=3, within [1,10] ✓
            settings.WarmStarting = true;
            solver.UpdateLinearCost(new double[] { -3.0 });
            solver.Solve();
            double x2  = solver.Solution.X[0];
            bool   ok2 = Math.Abs(x2 - 3.0) < 1e-3;

            if (ok2)
                Debug.Log($"[OsqpSmokeTest] PASS (warm-start) — x = {x2:F6}  (expected 3.0)");
            else
                Debug.LogError($"[OsqpSmokeTest] FAIL (warm-start) — x = {x2:F6}  (expected 3.0)");

            return ok && ok2;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[OsqpSmokeTest] EXCEPTION: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }
}