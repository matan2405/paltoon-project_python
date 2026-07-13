// Longitudinal DMPC Nash Equilibrium solver — Iterative Best Response (Li et al. 2019).
//
// Problem (per player, OSQP canonical form  min 0.5·u'Pu + q'u):
//   Player 1:  P1 = 2(H'Q1H + R1·I)      q1 = -2(H'Q1(r1-z_free) - H'Q1H·u2_k)
//   Player 2:  P2 = 2(α·H'Q1H + R2·I)    q2 = -2α(H'Q1(r2-z_free) - H'Q1H·u1_new)
//   α = 1/(1+λ)  — Pustilnik & Borrelli 2025
//
// Constraint matrix A_con  (2·Nu × Nu):
//   Rows 0..Nu-1     : I_Nu               → box  l_box ≤ u ≤ u_box
//   Row  Nu          : [1, 0, …, 0]       → jerk from prev:  u[0] - u_prev ∈ [-Δu, Δu]
//   Rows Nu+k (k≥1)  : -1 at col k-1, +1 at col k  → u[k]-u[k-1] ∈ [-Δu, Δu]
//
// Factory: call LongitudinalNashSolver.Create(w, t, vehicle) — linearises at vehicle's current vx.
using System;
using OsqpSharp;
using UnityEngine;

namespace NashPlatoon
{
    public sealed class LongitudinalNashSolver : NashSolverBase
    {
        // nx=2 states [x, vx], nz=2 outputs [x, vx] (C=I₂), nu_ctrl=1 scalar input.
        const int NxLong = 2;

        // ── OSQP instances ────────────────────────────────────────────────────
        OsqpSolver   _s1;         // Player 1 (λ-independent)
        OsqpSolver[] _s2;         // Player 2 — one per LambdaLevels entry (11 for long)

        // ── Constraint matrix (shared sparsity for all solvers) ───────────────
        double[] _Adata; int[] _Ai, _Ap;
        int _mCon;                                       // = 2 * Nu

        // ── Previous controls (for jerk-from-prev bound) ─────────────────────
        double _u1Prev, _u2Prev;

        // ── Warm-start horizon shifts ─────────────────────────────────────────
        double[]   _ws1;          // length Nu
        double[][] _ws2;          // [LambdaLevels.Length][Nu]

        // ── Optional SE-kernel GP risk modifier ───────────────────────────────
        SEKernelGP _seGp;
        public void SetGpModifier(SEKernelGP gp) => _seGp = gp;

        // ── Linearisation cache ────────────────────────────────────────────────
        float _vxCache = -1f;

        // ── Physical box bounds (updated each Nash step from vehicle model) ────
        float _u1Min, _u1Max, _u2Min, _u2Max;

        // ── Statistics (read-only) ────────────────────────────────────────────
        public int   LastIbrIters { get; private set; }
        public bool  LastConverged { get; private set; }
        public int   LastLambdaIdx { get; private set; }

        // ── Construction ─────────────────────────────────────────────────────

        LongitudinalNashSolver(NashWeightsConfig w, SimTimingConfig t) : base(w, t)
        {
            Nx = NxLong;
            Nz = 2;   // C = I₂ → outputs = states

            // Size arrays from the virtual property (LongLambdaLevels, 11 entries)
            _s2  = new OsqpSolver[LambdaLevels.Length];
            _ws1 = new double[Nu];
            _ws2 = new double[LambdaLevels.Length][];
            for (int i = 0; i < LambdaLevels.Length; i++)
                _ws2[i] = new double[Nu];
        }

        public static LongitudinalNashSolver Create(
            NashWeightsConfig w, SimTimingConfig t,
            AdvancedBicycleModel vehicle)
        {
            var s = new LongitudinalNashSolver(w, t);
            // Initialise physical bounds to config defaults until first UpdatePhysicalBounds call
            s._u1Min = w.U1Min; s._u1Max = w.U1Max;
            s._u2Min = w.U2Min; s._u2Max = w.U2Max;
            s.InitDoubleIntegratorC();              // builds C only
            s.UpdateLinearization(vehicle, t.NashDtLong);   // builds A,B from physics
            return s;
        }

        // ── State-space initialisation ────────────────────────────────────────
        // Only C is fixed (output = state). A and B are built in Relinearize from
        // vehicle.GetLongJacobian() so the solver matches the actual physics
        // (drag damping in A[1,1]) — mirrors the lateral solver's init pattern.
        void InitDoubleIntegratorC()
        {
            C = new double[] { 1, 0, 0, 1 };            // 2×2 = I₂
        }

        // ── NashSolverBase override ───────────────────────────────────────────

        public override void SetupSolvers(double[] Ad, double[] Bd)
        {
            A = MatrixOps.Copy(Ad);
            B = MatrixOps.Copy(Bd);
            // C stays I₂

            RebuildPredictionMatrices();
            RebuildCostMatrices();
            ApplyGpRiskHessian();      // optional: HQ1H_risk = (I+4γ·HQ1H·Σ)·HQ1H
            BuildConstraintMatrix();
            InitOsqpSolvers();
        }

        // ── Horizon weighting (Flad et al. 2017) ─────────────────────────────
        // Tracking cost scales from 1× at step 0 to wQ× at step Np-1.
        // wQ=2 for longitudinal (conservative: gap errors at 2 s still meaningful).
        protected override void FillQ1Block(double[] Q1full, int rows, int i, bool terminal)
        {
            double qp = terminal ? W.Q_pos_terminal : W.Q_pos;
            double qv = terminal ? W.Q_vel_terminal : W.Q_vel;
            if (!terminal && Np > 1)
            {
                double scale = 1.0 + (i / (double)(Np - 1)) * (2.0 - 1.0);
                qp *= scale;
                qv *= scale;
            }
            Q1full[(i * Nz + 0) * rows + (i * Nz + 0)] = qp;
            Q1full[(i * Nz + 1) * rows + (i * Nz + 1)] = qv;
        }

        /// Re-linearise with vehicle physical Jacobian when vx drifts > 2.0 m/s.
        /// Returns true if a rebuild was triggered.
        /// 2.0 m/s threshold mirrors LateralNashSolver: vx oscillates ±0.5 m/s
        /// between Nash steps at high speed → tighter thresholds trigger frequent
        /// SetupSolvers (re-factorisation of 12 OSQP problems) and break warm-start.
        public bool UpdateLinearization(AdvancedBicycleModel vehicle, float dt)
        {
            float vxNow = vehicle.GetVx();
            if (Mathf.Abs(vxNow - _vxCache) < 2.0f && _vxCache >= 0f) return false;
            _vxCache = vxNow;
            Relinearize(vehicle, dt);
            return true;
        }

        void Relinearize(AdvancedBicycleModel vehicle, float dt)
        {
            // GetLongJacobian returns Bc=[0,1] (u = a_des [m/s²], Belousov Eq. 3.11a).
            // Bc is re-asserted here as a contract guard: U1Min/U1Max/AccelerationToInputs
            // all assume m/s² units, and a force-unit Bc would collapse HQ1H by (1/mass)².
            var (Ac2d, Bc1d) = vehicle.GetLongJacobian();
            double[] Ac = { Ac2d[0,0], Ac2d[0,1], Ac2d[1,0], Ac2d[1,1] };
            double[] Bc = { 0.0, 1.0 };   // contract guard — matches GetLongJacobian Bc[1]
            var (Ad, Bd) = MatrixOps.ZOH(Ac, Bc, NxLong, 1, dt);
            SetupSolvers(Ad, Bd);
        }

        void ApplyGpRiskHessian()
        {
            if (_seGp == null || W.GammaRisk <= 0.0) return;
            HQ1H = _seGp.ModifyHessian(HQ1H, Nu, T.NashDtLong, W.GammaRisk);
        }

        // ── Constraint matrix ─────────────────────────────────────────────────

        void BuildConstraintMatrix()
        {
            _mCon = 2 * Nu;
            var Adense = new double[_mCon * Nu];

            // Rows 0..Nu-1: identity (box)
            for (int k = 0; k < Nu; k++)
                Adense[k * Nu + k] = 1.0;

            // Row Nu: picks u[0] for jerk-from-prev bound
            Adense[Nu * Nu + 0] = 1.0;

            // Rows Nu+k (k=1..Nu-1): u[k] − u[k-1]
            for (int k = 1; k < Nu; k++)
            {
                Adense[(Nu + k) * Nu + (k - 1)] = -1.0;
                Adense[(Nu + k) * Nu +  k]      =  1.0;
            }

            (_Adata, _Ai, _Ap) = MatrixOps.DenseToCSC(Adense, _mCon, Nu);
        }

        // ── OSQP solver initialisation ────────────────────────────────────────

        void InitOsqpSolvers()
        {
            DisposeOsqp();

            // Player 1 — P1 = 2*(HQ1H + R1·I)
            double[] P1 = BuildPMatrix((double)W.R1_long, 1.0);
            var (P1d, P1i, P1p) = MatrixOps.DenseSymToUpperCsc(P1, Nu);

            double[] q0 = new double[Nu];
            double[] l1 = new double[_mCon], u1 = new double[_mCon];
            FillBounds(l1, u1, W.U1Min, W.U1Max,
                       W.Du1Max * T.NashDtLong, 0f);

            _s1 = new OsqpSolver(SolverSettings());
            _s1.Setup(Nu, _mCon, P1d, P1i, P1p, q0, _Adata, _Ai, _Ap, l1, u1);

            // Player 2 — one solver per lambda level
            for (int i = 0; i < LambdaLevels.Length; i++)
            {
                double alpha = PustilnikAlpha(LambdaLevels[i]);
                double[] P2 = BuildPMatrix((double)W.R2_long, alpha);
                var (P2d, P2i, P2p) = MatrixOps.DenseSymToUpperCsc(P2, Nu);

                double[] l2 = new double[_mCon], u2 = new double[_mCon];
                FillBounds(l2, u2, W.U2Min, W.U2Max,
                           W.Du2Max * T.NashDtLong, 0f);

                _s2[i] = new OsqpSolver(SolverSettings());
                _s2[i].Setup(Nu, _mCon, P2d, P2i, P2p, q0, _Adata, _Ai, _Ap, l2, u2);
            }
        }

        // P_osqp = 2·(α·HQ1H + R·I)   (OSQP minimises 0.5·u'Pu → factor of 2)
        double[] BuildPMatrix(double R, double alpha)
        {
            double[] P = MatrixOps.Scale(HQ1H, 2.0 * alpha);
            MatrixOps.AddDiagInPlace(P, Nu, 2.0 * R);
            MatrixOps.SymmetrizeInPlace(P, Nu);
            return P;
        }

        // ── IBR solve ─────────────────────────────────────────────────────────

        public override (float u1, float u2) SolveNashEquilibrium(
            float[] x0, float[] r1f, float[] r2f, float lambda)
        {
            int    lamIdx = FindLambdaIdx(lambda);
            double alpha  = PustilnikAlpha(LambdaLevels[lamIdx]);

            // Free response  z_free = Umat · x0
            double[] x0d   = ToDouble(x0, Nx);
            double[] zFree = MatrixOps.MV(Umat, Np * Nz, Nx, x0d);

            // r1, r2 as double[]
            int rLen = Np * Nz;
            double[] r1 = ToDouble(r1f, rLen);
            double[] r2 = ToDouble(r2f, rLen);

            // HtQ1·(r - z_free) — constant within the IBR loop
            double[] HQ1r1 = MatrixOps.MV(HtQ1, Nu, rLen, MatrixOps.Sub(r1, zFree));
            double[] HQ1r2 = MatrixOps.MV(HtQ1, Nu, rLen, MatrixOps.Sub(r2, zFree));

            // Update jerk-from-prev bounds using physical box limits set by UpdatePhysicalBounds
            float du1 = W.Du1Max * T.NashDtLong;
            float du2 = W.Du2Max * T.NashDtLong;
            UpdateBoundsForCall(_s1, _u1Min, _u1Max, du1, (float)_u1Prev);
            UpdateBoundsForCall(_s2[lamIdx], _u2Min, _u2Max, du2, (float)_u2Prev);

            // Warm-start: inject shifted-horizon solution from previous Nash step
            InjectWarmStart(_s1,       _ws1);
            InjectWarmStart(_s2[lamIdx], _ws2[lamIdx]);

            // ── IBR loop ──────────────────────────────────────────────────────
            double[] u1k = MatrixOps.Copy(_ws1);
            double[] u2k = MatrixOps.Copy(_ws2[lamIdx]);
            double[] u1New = u1k, u2New = u2k;
            bool converged = false;

            int   maxIter = W.IbrMaxIter;
            float tol     = W.IbrTol;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // Player 1: fix u2k  →  q1 = -2·(HQ1r1 - HQ1H·u2k)
                double[] q1 = Gradient(HQ1r1, u2k, 1.0);
                _s1.UpdateLinearCost(q1);
                var sol1 = _s1.Solve();
                if (!IsOk(sol1)) break;
                u1New = MatrixOps.Copy(sol1.X);

                // Player 2: fix u1New  →  q2 = -2α·(HQ1r2 - HQ1H·u1New)
                double[] q2 = Gradient(HQ1r2, u1New, alpha);
                _s2[lamIdx].UpdateLinearCost(q2);
                var sol2 = _s2[lamIdx].Solve();
                if (!IsOk(sol2)) break;
                u2New = MatrixOps.Copy(sol2.X);

                double delta = MatrixOps.MaxAbs2(
                    MatrixOps.Sub(u1New, u1k),
                    MatrixOps.Sub(u2New, u2k));
                u1k = u1New; u2k = u2New;

                LastIbrIters = iter + 1;
                if (delta < tol) { converged = true; break; }
            }

            LastConverged = converged;
            LastLambdaIdx = lamIdx;

            // Horizon-shift warm starts for the next Nash step
            _ws1         = MatrixOps.ShiftLeft(u1New);
            _ws2[lamIdx] = MatrixOps.ShiftLeft(u2New);

            _u1Prev = u1New[0];
            _u2Prev = u2New[0];

            return ((float)u1New[0], (float)u2New[0]);
        }

        // ── Public helpers ────────────────────────────────────────────────────

        public void Reset()
        {
            _u1Prev = 0; _u2Prev = 0;
            Array.Clear(_ws1, 0, _ws1.Length);
            for (int i = 0; i < _ws2.Length; i++)
                Array.Clear(_ws2[i], 0, _ws2[i].Length);
        }

        /// Seed solver state from current ego acceleration (mirrors LateralNashSolver.WarmStartFromDriverState).
        /// Call every Nash tick during Approach/GapSearch so _u1Prev, _u2Prev, and the horizon
        /// warm-starts all reflect the physical vehicle state before the first Merge step.
        public void WarmStartPrev(float ax)
        {
            _u1Prev = ax;
            _u2Prev = ax;
            for (int k = 0; k < Nu; k++)
                _ws1[k] = ax;
            for (int i = 0; i < _ws2.Length; i++)
                for (int k = 0; k < Nu; k++)
                    _ws2[i][k] = ax;
        }

        /// Hot-path R2 update: rebuild P2 for all λ levels with new R2 value.
        /// Uses osqp_update_data_mat — no re-factorisation required.
        public void UpdateR2(float r2New)
        {
            for (int i = 0; i < LambdaLevels.Length; i++)
            {
                double alpha = PustilnikAlpha(LambdaLevels[i]);
                double[] P2  = BuildPMatrix((double)r2New, alpha);
                var (P2d, _, _) = MatrixOps.DenseSymToUpperCsc(P2, Nu);
                _s2[i].UpdateP(P2d);
            }
        }

        // ── IDisposable ───────────────────────────────────────────────────────

        public override void Dispose()
        {
            DisposeOsqp();
            GC.SuppressFinalize(this);
        }

        ~LongitudinalNashSolver() => Dispose();

        // ── Private helpers ───────────────────────────────────────────────────

        // q = -2·scale·(HQ1r - HQ1H·u_other)
        double[] Gradient(double[] HQ1r, double[] uOther, double scale)
        {
            double[] HQ1H_u = MatrixOps.MV(HQ1H, Nu, Nu, uOther);
            var q = new double[Nu];
            double s2 = -2.0 * scale;
            for (int j = 0; j < Nu; j++)
                q[j] = s2 * (HQ1r[j] - HQ1H_u[j]);
            return q;
        }

        void FillBounds(double[] l, double[] u, float uMin, float uMax,
                        float duStep, float uPrev)
        {
            for (int k = 0; k < Nu; k++) { l[k] = uMin; u[k] = uMax; }
            l[Nu] = uPrev - duStep;
            u[Nu] = uPrev + duStep;
            for (int k = 1; k < Nu; k++) { l[Nu + k] = -duStep; u[Nu + k] = duStep; }
        }

        void UpdateBoundsForCall(OsqpSolver solver,
                                 float uMin, float uMax, float duStep, float uPrev)
        {
            var l = new double[_mCon]; var u = new double[_mCon];
            FillBounds(l, u, uMin, uMax, duStep, uPrev);
            solver.UpdateBounds(l, u);
        }

        // Tighten box bounds using physical vehicle limits (called each Nash step before IBR).
        // Config bounds U1Min/U1Max and U2Min/U2Max act as hard safety ceilings;
        // physics bounds further restrict within them based on current vx.
        // Stores the result so SolveNashEquilibrium uses physical bounds, not config defaults.
        public void UpdatePhysicalBounds(float aMin, float aMax, float du1, float du2)
        {
            _u1Min = Mathf.Max(W.U1Min, aMin);
            _u1Max = Mathf.Min(W.U1Max, aMax);
            _u2Min = Mathf.Max(W.U2Min, aMin);
            _u2Max = Mathf.Min(W.U2Max, aMax);

            UpdateBoundsForCall(_s1, _u1Min, _u1Max, du1, (float)_u1Prev);
            for (int i = 0; i < _s2.Length; i++)
                UpdateBoundsForCall(_s2[i], _u2Min, _u2Max, du2, (float)_u2Prev);
        }

        static void InjectWarmStart(OsqpSolver solver, double[] ws)
        {
            if (solver.Solution.X.Length == ws.Length)
                Array.Copy(ws, solver.Solution.X, ws.Length);
        }

        static bool IsOk(OsqpSolution sol) =>
            sol.Status == OsqpStatus.Solved ||
            sol.Status == OsqpStatus.SolvedInaccurate;

        static OsqpSettings SolverSettings() => new OsqpSettings
        {
            WarmStarting      = true,
            MaxIter           = 1000,
            AbsoluteTolerance = 1e-4,
            RelativeTolerance = 1e-4,
            Verbose           = false,
            Polishing         = false,
        };

        static double[] ToDouble(float[] src, int len)
        {
            var dst = new double[len];
            for (int i = 0; i < len; i++) dst[i] = src[i];
            return dst;
        }

        void DisposeOsqp()
        {
            _s1?.Dispose(); _s1 = null;
            for (int i = 0; i < _s2.Length; i++) { _s2[i]?.Dispose(); _s2[i] = null; }
        }
    }
}