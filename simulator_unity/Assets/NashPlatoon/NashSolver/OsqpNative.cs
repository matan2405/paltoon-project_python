// Raw P/Invoke bindings for OSQP v1.0.0  (libosqp.dll / libosqp.so)
// Source-of-truth: github.com/sfiruch/osqp-net  (ported from LibraryImport → DllImport)
//
// CRITICAL — all integer fields are OSQPInt = int64_t  (OSQP_USE_LONG build):
//   • CSC index arrays: long[]  (not int[])
//   • All struct fields: long   (not int, even for C enums — padded to 8 B on x64)
//
// Requires: Player Settings → Other Settings → Allow 'unsafe' Code = true
using System;
using System.Runtime.InteropServices;

namespace OsqpSharp
{
    // ── Structs ───────────────────────────────────────────────────────────

    /// CSC sparse matrix descriptor — pointers owned by managed caller, owned=0.
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NativeCscMatrix
    {
        public long    m;       // rows
        public long    n;       // columns
        public long*   p;       // column pointers (size n+1)
        public long*   i;       // row indices
        public double* x;       // values
        public long    nzmax;
        public long    nz;      // -1 for CSC
        public long    owned;   // 0 = caller owns memory, OSQP will NOT free
    }

    /// Solver settings — all enum/bool fields declared as long to match int64_t ABI.
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeSettings
    {
        public long   Device;
        public long   LinsysSolver;       // enum linsys_solver_type (padded to 8 B)
        public long   AllocateSolution;
        public long   Verbose;
        public long   ProfilerLevel;
        public long   WarmStarting;
        public long   Scaling;
        public long   Polishing;
        public double Rho;
        public long   RhoIsVec;
        public double Sigma;
        public double Alpha;
        public long   CgMaxIter;
        public long   CgTolReduction;
        public double CgTolFraction;
        public long   CgPrecond;          // enum osqp_precond_type (padded to 8 B)
        public long   AdaptiveRho;
        public long   AdaptiveRhoInterval;
        public double AdaptiveRhoFraction;
        public double AdaptiveRhoTolerance;
        public long   MaxIter;
        public double EpsAbs;
        public double EpsRel;
        public double EpsPrimInf;
        public double EpsDualInf;
        public long   ScaledTermination;
        public long   CheckTermination;
        public long   CheckDualgap;
        public double TimeLimit;
        public double Delta;
        public long   PolishRefineIter;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NativeSolution
    {
        public double* X;           // primal (size n)
        public double* Y;           // dual   (size m)
        public double* PrimInfCert;
        public double* DualInfCert;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NativeInfo
    {
        public fixed byte StatusStr[32];
        public long   StatusVal;
        public long   StatusPolish;
        public double ObjVal, DualObjVal, PrimRes, DualRes, DualityGap;
        public long   Iter, RhoUpdates;
        public double RhoEstimate;
        public double SetupTime, SolveTime, UpdateTime, PolishTime, RunTime;
        public double PrimDualInt, RelKktError;
    }

    /// Top-level solver struct — OSQPSolver in C; accessed via pointer after osqp_setup.
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NativeSolverHandle
    {
        public NativeSettings* Settings;
        public NativeSolution* Solution;
        public NativeInfo*     Info;
        public void*           Work;     // opaque internal workspace
    }

    // ── P/Invoke ──────────────────────────────────────────────────────────

    internal static unsafe class OsqpApi
    {
        const string Lib = "libosqp";

        /// Fill *s with safe defaults before modifying specific fields.
        [DllImport(Lib, EntryPoint = "osqp_set_default_settings")]
        public static extern void SetDefaultSettings(NativeSettings* s);

        /// Allocate and initialise solver.  Writes OSQPSolver* into *solverOut.
        /// Returns 0 on success, non-zero on error.
        [DllImport(Lib, EntryPoint = "osqp_setup")]
        public static extern long Setup(
            IntPtr*          solverOut,
            NativeCscMatrix* P,
            double*          q,
            NativeCscMatrix* A,
            double*          l,
            double*          u,
            long             m,
            long             n,
            NativeSettings*  settings);

        /// Solve the currently loaded QP.
        [DllImport(Lib, EntryPoint = "osqp_solve")]
        public static extern long Solve(IntPtr solver);

        /// Free all solver memory.
        [DllImport(Lib, EntryPoint = "osqp_cleanup")]
        public static extern long Cleanup(IntPtr solver);

        /// Provide a warm-start point (either x or y may be null to skip).
        [DllImport(Lib, EntryPoint = "osqp_warm_start")]
        public static extern long WarmStart(IntPtr solver, double* x, double* y);

        /// Update q, l and/or u in-place (any pointer may be null to skip).
        [DllImport(Lib, EntryPoint = "osqp_update_data_vec")]
        public static extern long UpdateVec(
            IntPtr  solver,
            double* q,
            double* l,
            double* u);

        /// Update non-zero values of P and/or A (sparsity pattern must be identical).
        /// Pass null + 0 for the matrix you are not updating.
        [DllImport(Lib, EntryPoint = "osqp_update_data_mat")]
        public static extern long UpdateMat(
            IntPtr  solver,
            double* Px,    long* PxIdx, long Pn,
            double* Ax,    long* AxIdx, long An);

        /// Returns a pointer to a static version string (do not free).
        [DllImport(Lib, EntryPoint = "osqp_version")]
        public static extern IntPtr Version();
    }
}
