// Lateral DMPC Nash Equilibrium solver — Iterative Best Response (Li et al. 2019).
//
// State:   x = [y, ẏ, ψ, ψ̇]                      nx = 4
// Output:  z = [y, ψ]    (C selects indices 0, 2) nz = 2
// Control: u = δ (front steering angle, rad)      nu = 1 per step (× Nu horizon)
//
// Problem (per player, OSQP canonical form  min 0.5·u'Pu + q'u):
//   Player 1:  P1 = 2(H'Q1H + R1·I)      q1 = -2(H'Q1(r1-z_free) - H'Q1H·u2_k)
//   Player 2:  P2 = 2(α·H'Q1H + R2·I)    q2 = -2α(H'Q1(r2-z_free) - H'Q1H·u1_new)
//   α = 1/(1+λ)  — Pustilnik & Borrelli 2025
//
// Constraint matrix (2·Nu × Nu): box on δ + rate |dδ| ≤ DDeltaMax.
//
// Continuous-time A_c, B_c from linearised bicycle model at current vx;
// ZOH-discretised at dt = NashDtLat (20 Hz). Re-linearise via SetupSolvers
// when vx drifts more than LAT_NASH_REBUILD_DVX (Coriolis sensitivity).
//
// Factory: LateralNashSolver.Create(w, t, vehicle) — linearises at vehicle's current vx.
using System;
using OsqpSharp;
using UnityEngine;

namespace NashPlatoon
{
    public sealed class LateralNashSolver : NashSolverBase
    {
        const int NxLat = 4;
        const int NzLat = 2;

        // ── OSQP instances ────────────────────────────────────────────────────
        OsqpSolver   _s1;
        OsqpSolver[] _s2;    // one per LatLambdaLevels entry (7)

        // ── Constraint matrix ─────────────────────────────────────────────────
        double[] _Adata; int[] _Ai, _Ap;
        int _mCon;

        // ── Previous controls (rate limit) ───────────────────────────────────────
        double _u1Prev, _u2Prev;

        // ── Measured lateral acceleration (direct, not via delta) ─────────────
        float _aLatPrev;    // a_lat[k-1] — for jerk = Δa_lat/dt
        float _aLatCurrent; // updated via SetALat() each Nash tick before SolveNashEquilibrium

        // ── Warm-start horizon shifts ─────────────────────────────────────────
        double[]   _ws1;
        double[][] _ws2;

        // ── Runtime Q overrides (set by UpdateQ1; -1 = use W defaults) ──────────
        float _qyOverride       = -1f;
        float _qpsiOverride     = -1f;
        float _qyTermOverride   = -1f;
        float _qpsiTermOverride = -1f;

        // ── Current adaptive R2 (set by UpdateR2; used by UpdateQ1 to avoid reset) ──
        float _r2Current = -1f;

        // ── Linearisation cache ───────────────────────────────────────────────
        float _vxCache = -1f;

        // ── R79 Category C transient jerk tracker (system player only) ────────
        // Tracks how long system jerk has exceeded LatJerkMaxHuman.
        // Resets immediately when jerk drops back to ≤ LatJerkMaxHuman.
        float _highJerkTime = 0f;

        // ── Optional risk-aware Hessian modifier (SE-kernel GP) ───────────────
        SEKernelGP _seGp;
        public void SetGpModifier(SEKernelGP gp) => _seGp = gp;

        // ── Stats ─────────────────────────────────────────────────────────────
        public int   LastIbrIters { get; private set; }
        public bool  LastConverged { get; private set; }
        public int   LastLambdaIdx { get; private set; }

        // Override lambda levels — lateral uses 7-level set
        protected override double[] LambdaLevels => LatLambdaLevels;

        // ── Construction ─────────────────────────────────────────────────────

        LateralNashSolver(NashWeightsConfig w, SimTimingConfig t) : base(w, t)
        {
            Nx  = NxLat;
            Nz  = NzLat;

            _s2  = new OsqpSolver[LambdaLevels.Length];
            _ws1 = new double[Nu];
            _ws2 = new double[LambdaLevels.Length][];
            for (int i = 0; i < LambdaLevels.Length; i++)
                _ws2[i] = new double[Nu];
        }

        public static LateralNashSolver Create(
            NashWeightsConfig w, SimTimingConfig t,
            AdvancedBicycleModel vehicle)
        {
            var s = new LateralNashSolver(w, t);
            s.InitBicycleC();
            s.UpdateLinearization(vehicle);   // builds A,B and all solvers
            return s;
        }

        // C selects [y, ψ] from state [y, ẏ, ψ, ψ̇]
        void InitBicycleC()
        {
            C = new double[NzLat * NxLat];
            C[0 * NxLat + 0] = 1.0;   // y
            C[1 * NxLat + 2] = 1.0;   // ψ
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// Rebuild Ac/Bc from vehicle.GetLatJacobian() and refresh all prediction/cost/OSQP
        /// matrices. Returns true if a rebuild was triggered (vx drifted > 2 m/s).
        public bool UpdateLinearization(AdvancedBicycleModel vehicle)
        {
            float vx = vehicle.GetVx();
            // 2.0 m/s threshold: at high speed vx oscillates ±0.5 m/s between Nash steps →
            // frequent SetupSolvers calls broke warm-start consistency → oscillation.
            if (Mathf.Abs(vx - _vxCache) < 2.0f && _vxCache > 0f) return false;
            _vxCache = vx;

            var (Ac2d, Bc1d) = vehicle.GetLatJacobian();
            double[] Ac = ToDoubleRowMajor(Ac2d, NxLat);
            double[] Bc = ToDouble(Bc1d, NxLat);
            var (Ad, Bd) = MatrixOps.ZOH(Ac, Bc, NxLat, 1, T.NashDtLat);
            SetupSolvers(Ad, Bd);
            return true;
        }

        static double[] ToDoubleRowMajor(float[,] m, int n)
        {
            var d = new double[n * n];
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    d[r * n + c] = m[r, c];
            return d;
        }

        static double[] ToDouble(float[] src, int len)
        {
            var d = new double[len];
            for (int i = 0; i < len; i++) d[i] = src[i];
            return d;
        }

        public override void SetupSolvers(double[] Ad, double[] Bd)
        {
            A = MatrixOps.Copy(Ad);
            B = MatrixOps.Copy(Bd);

            RebuildPredictionMatrices();
            RebuildCostMatrices();
            ApplyGpRiskHessian();      // optional: HQ1H_risk = (I+4γ·HQ1H·Σ)·HQ1H
            BuildConstraintMatrix();
            InitOsqpSolvers();
            // Re-apply runtime overrides that were active before re-linearisation.
            // InitOsqpSolvers builds P2 from W.R2_lat; restore the adaptive R2/Q1 state
            // so a vx-triggered rebuild does not silently discard coordinator updates.
            if (_r2Current > 0f)
                UpdateR2(_r2Current);
            if (_qyOverride > 0f)
                UpdateQ1(_qyOverride, _qpsiOverride, _qyTermOverride, _qpsiTermOverride);
        }

        // ── Q1 block: [Q_y, Q_psi] with horizon weighting (Flad et al. 2017) ──
        // Tracking cost scales from 1× at step 0 to wQ× at step Np-1 so the solver
        // penalises far-horizon deviations more heavily than near-horizon ones.
        // Uses _qyOverride / _qpsiOverride when set (>0), otherwise falls back to W defaults.
        protected override void FillQ1Block(double[] Q1full, int rows, int i, bool terminal)
        {
            double qy   = terminal
                ? (_qyTermOverride   > 0f ? _qyTermOverride   : W.Q_y_terminal)
                : (_qyOverride       > 0f ? _qyOverride       : W.Q_y);
            double qpsi = terminal
                ? (_qpsiTermOverride > 0f ? _qpsiTermOverride : W.Q_psi_terminal)
                : (_qpsiOverride     > 0f ? _qpsiOverride     : W.Q_psi);
            if (!terminal && Np > 1)
            {
                double scale = 1.0 + (i / (double)(Np - 1)) * (1.5 - 1.0);
                qy   *= scale;
                qpsi *= scale;
            }
            Q1full[(i * Nz + 0) * rows + (i * Nz + 0)] = qy;
            Q1full[(i * Nz + 1) * rows + (i * Nz + 1)] = qpsi;
        }

        // ── Apply SE-kernel GP risk Hessian modification ──────────────────────
        // HQ1H_risk = (I + 4·γ̃·HQ1H·Σ) · HQ1H,  γ̃ = γ / trace(HQ1H)
        void ApplyGpRiskHessian()
        {
            if (_seGp == null || W.GammaRisk <= 0.0) return;
            HQ1H = _seGp.ModifyHessian(HQ1H, Nu, T.NashDtLat, W.GammaRisk);
        }

        // ── Constraint matrix ─────────────────────────────────────────────────

        void BuildConstraintMatrix()
        {
            // Block 0 (Nu rows):   box on δ[k]
            // Block 1 (Nu rows):   rate  |δ[k]-δ[k-1]| ≤ DDeltaMax
            // ISO 11270 §5.4 jerk limit enforced as soft cost (R_jerk·L'L in P),
            // not as a hard constraint — hard jerk at dt=0.05s gives duJerk≈0.000625 rad,
            // which is 7× tighter than DDeltaMax and causes QP infeasibility on any
            // meaningful lane correction, returning garbage from OSQP.
            _mCon = 2 * Nu;
            var Adense = new double[_mCon * Nu];

            // Block 0: box rows — identity
            for (int k = 0; k < Nu; k++)
                Adense[k * Nu + k] = 1.0;

            // Block 1: rate rows — |δ[k]-δ[k-1]| ≤ DDeltaMax
            Adense[Nu * Nu + 0] = 1.0;
            for (int k = 1; k < Nu; k++)
            {
                Adense[(Nu + k) * Nu + (k - 1)] = -1.0;
                Adense[(Nu + k) * Nu +  k]      =  1.0;
            }

            (_Adata, _Ai, _Ap) = MatrixOps.DenseToCSC(Adense, _mCon, Nu);
        }

        // ── OSQP initialisation ───────────────────────────────────────────────

        void InitOsqpSolvers()
        {
            DisposeOsqp();

            double[] q0 = new double[Nu];

            // Player 1
            double[] P1 = BuildPMatrix((double)W.R1_lat, 1.0);
            var (P1d, P1i, P1p) = MatrixOps.DenseSymToUpperCsc(P1, Nu);
            double[] l1 = new double[_mCon], u1 = new double[_mCon];
            FillBounds(l1, u1, W.DeltaMin, W.DeltaMax, DDeltaFromJerk(W.LatJerkMaxSystem, 22f), 0f);
            _s1 = new OsqpSolver(SolverSettings());
            _s1.Setup(Nu, _mCon, P1d, P1i, P1p, q0, _Adata, _Ai, _Ap, l1, u1);

            // Player 2 (one per λ)
            for (int i = 0; i < LambdaLevels.Length; i++)
            {
                double alpha = PustilnikAlpha(LambdaLevels[i]);
                double[] P2 = BuildPMatrix((double)W.R2_lat, alpha);
                var (P2d, P2i, P2p) = MatrixOps.DenseSymToUpperCsc(P2, Nu);

                double[] l2 = new double[_mCon], u2 = new double[_mCon];
                FillBounds(l2, u2, W.DeltaMin, W.DeltaMax, DDeltaFromJerk(W.LatJerkMaxHuman, 22f), 0f);

                _s2[i] = new OsqpSolver(SolverSettings());
                _s2[i].Setup(Nu, _mCon, P2d, P2i, P2p, q0, _Adata, _Ai, _Ap, l2, u2);
            }
        }

        double[] BuildPMatrix(double R, double alpha)
        {
            double[] P = MatrixOps.Scale(HQ1H, 2.0 * alpha);
            MatrixOps.AddDiagInPlace(P, Nu, 2.0 * R);
            MatrixOps.SymmetrizeInPlace(P, Nu);
            return P;
        }

        // ── IBR solve ─────────────────────────────────────────────────────────

        public void SetALat(float aLat) => _aLatCurrent = aLat;

        public override (float u1, float u2) SolveNashEquilibrium(
            float[] x0, float[] r1f, float[] r2f, float lambda)
        {
            float aLatCurrent = _aLatCurrent;
            int    lamIdx = FindLambdaIdx(lambda);
            double alpha  = PustilnikAlpha(LambdaLevels[lamIdx]);

            double[] x0d   = ToDouble(x0, Nx);
            double[] zFree = MatrixOps.MV(Umat, Np * Nz, Nx, x0d);

            int rLen = Np * Nz;
            double[] r1 = ToDouble(r1f, rLen);
            double[] r2 = ToDouble(r2f, rLen);

            double[] HQ1r1 = MatrixOps.MV(HtQ1, Nu, rLen, MatrixOps.Sub(r1, zFree));
            double[] HQ1r2 = MatrixOps.MV(HtQ1, Nu, rLen, MatrixOps.Sub(r2, zFree));

            float jerkSystem = SystemJerkLimit(aLatCurrent);
            UpdateBoundsForCall(_s1,
                W.DeltaMin, W.DeltaMax, DDeltaFromJerk(jerkSystem, _vxCache), (float)_u1Prev);
            UpdateBoundsForCall(_s2[lamIdx],
                W.DeltaMin, W.DeltaMax, DDeltaFromJerk(W.LatJerkMaxHuman, _vxCache), (float)_u2Prev);

            InjectWarmStart(_s1,        _ws1);
            InjectWarmStart(_s2[lamIdx], _ws2[lamIdx]);

            double[] u1k = MatrixOps.Copy(_ws1);
            double[] u2k = MatrixOps.Copy(_ws2[lamIdx]);
            double[] u1New = u1k, u2New = u2k;
            bool converged = false;

            int   maxIter = W.IbrMaxIter;
            float tol     = W.IbrTol;

            for (int iter = 0; iter < maxIter; iter++)
            {
                double[] q1 = Gradient(HQ1r1, u2k, 1.0);
                _s1.UpdateLinearCost(q1);
                var sol1 = _s1.Solve();
                if (!IsOk(sol1)) break;
                u1New = MatrixOps.Copy(sol1.X);

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

            _ws1         = MatrixOps.ShiftLeft(u1New);
            _ws2[lamIdx] = MatrixOps.ShiftLeft(u2New);

            _u1Prev = u1New[0];
            _u2Prev = u2New[0];

            return ((float)u1New[0], (float)u2New[0]);
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        public void Reset()
        {
            _u1Prev = 0; _u2Prev = 0; _aLatPrev = 0f; _highJerkTime = 0f;
            Array.Clear(_ws1, 0, _ws1.Length);
            for (int i = 0; i < _ws2.Length; i++)
                Array.Clear(_ws2[i], 0, _ws2[i].Length);
        }

        // Seed solver history from driver state just before Nash activation.
        // driverDelta: current wheel angle [rad] from ego.GetLambda()
        // aLatNow    : measured lateral acceleration [m/s²] from ego.GetALat()
        public void WarmStartFromDriverState(float driverDelta, float aLatNow)
        {
            _u1Prev       = driverDelta;
            _u2Prev       = driverDelta;
            _aLatPrev     = aLatNow;
            _highJerkTime = 0f;
            for (int k = 0; k < _ws1.Length; k++) _ws1[k] = driverDelta;
            for (int i = 0; i < _ws2.Length; i++)
                for (int k = 0; k < _ws2[i].Length; k++) _ws2[i][k] = driverDelta;
        }

        /// Hot-path R2 update: rebuild P2 for all λ levels with new R2 value.
        /// Uses osqp_update_data_mat — no re-factorisation required.
        public void UpdateR2(float r2New)
        {
            _r2Current = r2New;
            for (int i = 0; i < LambdaLevels.Length; i++)
            {
                double alpha = PustilnikAlpha(LambdaLevels[i]);
                double[] P2  = BuildPMatrix((double)r2New, alpha);
                var (P2d, _, _) = MatrixOps.DenseSymToUpperCsc(P2, Nu);
                _s2[i].UpdateP(P2d);
            }
        }

        /// Hot-path Q1 update: stores overrides, rebuilds HQ1H/HtQ1 and P1/P2 for all λ levels.
        /// Full RebuildCostMatrices is required because Q is baked into HQ1H.
        /// Guard with a threshold in the caller to avoid per-tick rebuilds.
        public void UpdateQ1(float qy, float qpsi, float qyTerm, float qpsiTerm)
        {
            _qyOverride       = qy;
            _qpsiOverride     = qpsi;
            _qyTermOverride   = qyTerm;
            _qpsiTermOverride = qpsiTerm;
            RebuildCostMatrices();
            ApplyGpRiskHessian();

            // Rebuild P1 (uses current R1_lat from W, Q encoded in HQ1H)
            double[] P1 = BuildPMatrix((double)W.R1_lat, 1.0);
            var (P1d, _, _) = MatrixOps.DenseSymToUpperCsc(P1, Nu);
            _s1.UpdateP(P1d);

            // Rebuild P2 with the current adaptive R2 (not the static W.R2_lat default)
            // so a Q1 rebuild cannot silently reset the R2 that UpdateR2 just applied.
            double r2 = _r2Current > 0f ? _r2Current : W.R2_lat;
            for (int i = 0; i < LambdaLevels.Length; i++)
            {
                double alpha = PustilnikAlpha(LambdaLevels[i]);
                double[] P2  = BuildPMatrix(r2, alpha);
                var (P2d, _, _) = MatrixOps.DenseSymToUpperCsc(P2, Nu);
                _s2[i].UpdateP(P2d);
            }
        }

        public override void Dispose()
        {
            DisposeOsqp();
            GC.SuppressFinalize(this);
        }

        ~LateralNashSolver() => Dispose();

        double[] Gradient(double[] HQ1r, double[] uOther, double scale)
        {
            double[] HQ1H_u = MatrixOps.MV(HQ1H, Nu, Nu, uOther);
            var q = new double[Nu];
            double s2 = -2.0 * scale;
            for (int j = 0; j < Nu; j++)
                q[j] = s2 * (HQ1r[j] - HQ1H_u[j]);
            return q;
        }

        // DDelta = LatJerkMax × NashDtLat / vx  (UN ECE-R79 jerk→rate conversion)
        float DDeltaFromJerk(float jerkMax, float vx) =>
            vx > 1f ? jerkMax * T.NashDtLat / vx : jerkMax * T.NashDtLat / 22f;

        // Returns allowed system jerk based on measured a_lat from physics.
        // actualJerk = |aLat[k] - aLat[k-1]| / NashDtLat  (direct measurement, no delta approximation)
        float SystemJerkLimit(float aLatCurrent)
        {
            float actualJerk = Mathf.Abs(aLatCurrent - _aLatPrev) / T.NashDtLat;
            _aLatPrev = aLatCurrent;

            if (actualJerk > W.LatJerkMaxHuman)
            {
                _highJerkTime += T.NashDtLat;
                return _highJerkTime <= W.LatJerkTransientSec
                    ? W.LatJerkMaxSystem
                    : W.LatJerkMaxHuman;
            }

            _highJerkTime = 0f;
            return W.LatJerkMaxSystem;
        }

        void FillBounds(double[] l, double[] u, float uMin, float uMax,
                        float duStep, float uPrev)
        {
            // Block 0: box
            for (int k = 0; k < Nu; k++) { l[k] = uMin; u[k] = uMax; }

            // Block 1: rate  |δ[k]-δ[k-1]| ≤ duStep
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
            MaxIter           = 600,    // LAT_NASH_MAX_ITER — matches Python config
            AbsoluteTolerance = 1e-3,   // LAT_NASH_EPS_ABS  — matches Python config
            RelativeTolerance = 1e-3,   // LAT_NASH_EPS_REL  — matches Python config
            Verbose           = false,
            Polishing         = false,
        };



        void DisposeOsqp()
        {
            _s1?.Dispose(); _s1 = null;
            for (int i = 0; i < _s2.Length; i++) { _s2[i]?.Dispose(); _s2[i] = null; }
        }
    }
}
