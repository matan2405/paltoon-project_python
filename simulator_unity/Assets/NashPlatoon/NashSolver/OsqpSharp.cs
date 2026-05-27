// Managed wrapper around native OSQP v1.0.0 (OsqpNative.cs / libosqp.dll).
// Public API matches the usage expected by NashSolverBase (Plans 6-8).
//
// Lifecycle:
//   Setup()              — first call or after problem structure changes
//   UpdateLinearCost(q)  — hot-path: change q without re-factorisation
//   UpdateBounds(l,u)    — hot-path: change constraint bounds
//   Solve()              — returns OsqpSolution (warm-starts automatically)
//   Dispose()            — frees native memory (also called by finalizer)
//
// Requires: Player Settings → Other Settings → Allow 'unsafe' Code = true
using System;

namespace OsqpSharp
{
    // ── Public types ──────────────────────────────────────────────────────

    public enum OsqpStatus
    {
        Solved                     = 1,
        SolvedInaccurate           = 2,
        PrimalInfeasible           = 3,
        PrimalInfeasibleInaccurate = 4,
        DualInfeasible             = 5,
        DualInfeasibleInaccurate   = 6,
        MaxIterReached             = 7,
        NonConvex                  = 9,
        TimeLimitReached           = 10,
        Unsolved                   = 11,
    }

    public class OsqpSettings
    {
        public bool   WarmStarting      = true;
        public int    MaxIter           = 1000;
        public double AbsoluteTolerance = 1e-5;
        public double RelativeTolerance = 1e-5;
        public double Rho               = 0.1;
        public double Sigma             = 1e-6;
        public double Alpha             = 1.6;
        public bool   Verbose           = false;
        public bool   Polishing         = false;
    }

    public class OsqpSolution
    {
        public double[]    X          = Array.Empty<double>();
        public double[]    Y          = Array.Empty<double>();
        public OsqpStatus  Status     = OsqpStatus.Unsolved;
        public double      ObjVal     = double.NaN;
        public int         Iterations;
    }

    // ── OsqpSolver ────────────────────────────────────────────────────────

    /// <summary>
    /// Thin managed wrapper around OSQP v1.0.0.
    /// Solves: min 0.5·x'Px + q'x  s.t.  l ≤ Ax ≤ u
    /// P: upper-triangular CSC (int[] indices, converted to long[] internally).
    /// A: general         CSC (int[] indices).
    /// </summary>
    public sealed unsafe class OsqpSolver : IDisposable
    {
        readonly OsqpSettings _s;
        IntPtr _handle;       // OSQPSolver* — set by osqp_setup, freed by osqp_cleanup
        int    _n, _m;

        // long[] copies of int[] CSC index arrays (OSQP v1 uses int64_t everywhere)
        long[] _Pp, _Pi, _Ap, _Ai;

        public OsqpSolution Solution { get; } = new OsqpSolution();

        public OsqpSolver(OsqpSettings settings = null)
        {
            _s = settings ?? new OsqpSettings();
        }

        // ── Setup ─────────────────────────────────────────────────────────

        /// Full setup from CSC-format arrays.  Re-call only when the sparsity
        /// pattern changes; use Update* for hot-path changes.
        public void Setup(int n, int m,
                          double[] Pdata, int[] Pi, int[] Pp,
                          double[] q,
                          double[] Adata, int[] Ai, int[] Ap,
                          double[] l,     double[] u)
        {
            // Free any existing solver
            if (_handle != IntPtr.Zero) { OsqpApi.Cleanup(_handle); _handle = IntPtr.Zero; }

            _n = n; _m = m;

            // Convert int[] → long[] (OSQP v1.0.0 uses int64_t for all indices)
            _Pp = ToLong(Pp); _Pi = ToLong(Pi);
            _Ap = ToLong(Ap); _Ai = ToLong(Ai);

            // Populate NativeSettings from managed settings
            NativeSettings ns;
            OsqpApi.SetDefaultSettings(&ns);
            WriteSettings(&ns);

            // Pin all arrays for the duration of the osqp_setup call.
            // OSQP v1 copies data internally → no need to keep them pinned after.
            fixed (double* pPd = Pdata, pAd = Adata, pQ = q, pL = l, pU = u)
            fixed (long*   pPp = _Pp,   pPi = _Pi,  pAp = _Ap, pAi = _Ai)
            {
                var P = new NativeCscMatrix
                {
                    m = n, n = n, x = pPd, i = pPi, p = pPp,
                    nzmax = Pdata.Length, nz = -1, owned = 0
                };
                var A = new NativeCscMatrix
                {
                    m = m, n = n, x = pAd, i = pAi, p = pAp,
                    nzmax = Adata.Length, nz = -1, owned = 0
                };

                IntPtr ptr = IntPtr.Zero;
                long err = OsqpApi.Setup(&ptr, &P, pQ, &A, pL, pU, m, n, &ns);

                if (err != 0 || ptr == IntPtr.Zero)
                    throw new InvalidOperationException(
                        $"osqp_setup failed (error {err}). " +
                        "Verify libosqp.dll is in Assets/Plugins/OSQP/ and " +
                        "configured for Windows x64 in the Inspector.");

                _handle = ptr;
            }
        }

        // ── Hot-path updates (no re-factorisation) ────────────────────────

        /// Change q (linear cost).  Solver re-uses its existing factorisation.
        public void UpdateLinearCost(double[] q)
        {
            if (_handle == IntPtr.Zero || q == null) return;
            fixed (double* pQ = q) OsqpApi.UpdateVec(_handle, pQ, null, null);
        }

        /// Change the non-zero values of P in-place (sparsity pattern must be identical).
        /// Pdata must contain the same number of entries as the original Pdata in Setup.
        public unsafe void UpdateP(double[] Pdata)
        {
            if (_handle == IntPtr.Zero || Pdata == null) return;
            fixed (double* pPd = Pdata)
                OsqpApi.UpdateMat(_handle, pPd, null, (long)Pdata.Length, null, null, 0L);
        }

        /// Change l and/or u (constraint bounds).
        public void UpdateBounds(double[] l, double[] u)
        {
            if (_handle == IntPtr.Zero) return;
            fixed (double* pL = l, pU = u)
                OsqpApi.UpdateVec(_handle, null, l != null ? pL : null, u != null ? pU : null);
        }

        // ── Solve ─────────────────────────────────────────────────────────

        public OsqpSolution Solve()
        {
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Call Setup() first.");

            // Warm-start with the previous solution (if available)
            if (_s.WarmStarting && Solution.X.Length == _n)
            {
                if (Solution.Y.Length == _m)
                {
                    fixed (double* px = Solution.X, py = Solution.Y)
                        OsqpApi.WarmStart(_handle, px, py);
                }
                else
                {
                    fixed (double* px = Solution.X)
                        OsqpApi.WarmStart(_handle, px, null);
                }
            }

            OsqpApi.Solve(_handle);

            // Read back solution from the native struct
            ReadSolution();
            return Solution;
        }

        // ── IDisposable ───────────────────────────────────────────────────

        public void Dispose()
        {
            if (_handle != IntPtr.Zero) {
                OsqpApi.Cleanup(_handle);
                _handle = IntPtr.Zero;
            }
        }

        ~OsqpSolver() => Dispose();

        // ── Private helpers ───────────────────────────────────────────────

        void WriteSettings(NativeSettings* ns)
        {
            ns->Verbose      = _s.Verbose      ? 1 : 0;
            ns->WarmStarting = 0;           // warm-start managed manually via osqp_warm_start
            ns->MaxIter      = _s.MaxIter;
            ns->EpsAbs       = _s.AbsoluteTolerance;
            ns->EpsRel       = _s.RelativeTolerance;
            ns->Rho          = _s.Rho;
            ns->Sigma        = _s.Sigma;
            ns->Alpha        = _s.Alpha;
            ns->Polishing    = _s.Polishing ? 1 : 0;
            ns->AllocateSolution = 1;      // ensure Solution pointer is populated
        }

        void ReadSolution()
        {
            var native = (NativeSolverHandle*)_handle.ToPointer();

            if (native->Solution != null && native->Solution->X != null)
            {
                if (Solution.X.Length != _n) Solution.X = new double[_n];
                if (Solution.Y.Length != _m) Solution.Y = new double[_m];
                for (int i = 0; i < _n; i++) Solution.X[i] = native->Solution->X[i];
                for (int i = 0; i < _m; i++) Solution.Y[i] = native->Solution->Y[i];
            }

            if (native->Info != null)
            {
                Solution.Status     = (OsqpStatus)(int)native->Info->StatusVal;
                Solution.ObjVal     = native->Info->ObjVal;
                Solution.Iterations = (int)native->Info->Iter;
            }
        }

        static long[] ToLong(int[] src)
        {
            var dst = new long[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = src[i];
            return dst;
        }
    }
}