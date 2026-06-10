// SEKernelGP — Squared-Exponential kernel GP for MA-IDM risk-aware Nash planning.
//
// Mirrors longitudinal_constrained_nash_solver.py lines 394-414:
//
//   Σ_ε[i,j] = σ_k² · exp(−(i−j)²·dt² / (2ℓ²))            (SE kernel, Eq. 9 of Zhang & Sun 2024)
//   γ̃        = γ_risk / trace(HQ1H)                       (Hessian-normalised risk gain)
//   M_risk   = 4·γ̃·HQ1H·Σ_ε
//   HQ1H_risk= (I + M_risk) · HQ1H                         (risk-aware Hessian)
//
// The modification inflates the Nash control cost along directions where the
// kernel predicts high-variance residual disturbance — making the planner act
// more conservatively when uncertainty about driver behaviour is high.
//
// Σ_ε is Nu×Nu (one entry per control-horizon step).
// HQ1H is Nu×Nu. All arithmetic is double precision.
//
// Used by LateralNashSolver / LongitudinalNashSolver via SetGpModifier(...).
using System;

namespace NashPlatoon
{
    public sealed class SEKernelGP
    {
        readonly double _sigmaK;
        readonly double _ell;

        // Per-Nu cache of the SE-kernel covariance matrix
        double[] _Sigma;
        int      _NuCache = -1;
        double   _dtCache = -1;

        public SEKernelGP(double sigmaK, double ell)
        {
            _sigmaK = sigmaK;
            _ell    = ell;
        }

        public SEKernelGP(NashWeightsConfig w) : this(w.SigmaK, w.Ell) { }

        /// <summary>
        /// Build (and cache) the SE-kernel covariance Σ_ε for control horizon length Nu
        /// and time step dt: Σ[i,j] = σ_k² · exp(−(i−j)²·dt² / (2·ℓ²)).
        /// </summary>
        public double[] BuildSEKernel(int Nu, double dt)
        {
            if (_Sigma != null && _NuCache == Nu && Math.Abs(_dtCache - dt) < 1e-9)
                return _Sigma;

            _NuCache = Nu;
            _dtCache = dt;
            _Sigma   = new double[Nu * Nu];

            double inv2L2 = 1.0 / (2.0 * _ell * _ell);
            double s2     = _sigmaK * _sigmaK;
            for (int i = 0; i < Nu; i++)
            {
                for (int j = 0; j < Nu; j++)
                {
                    double d = (i - j) * dt;
                    _Sigma[i * Nu + j] = s2 * Math.Exp(-d * d * inv2L2);
                }
            }
            return _Sigma;
        }

        /// <summary>
        /// Modify the Nash Hessian in place of a fresh copy:
        ///   HQ1H_risk = (I + 4·γ̃·HQ1H·Σ_ε) · HQ1H,  γ̃ = γ_risk / trace(HQ1H)
        /// </summary>
        public double[] ModifyHessian(double[] HQ1H, int Nu, double dt, double gammaRisk)
        {
            if (gammaRisk <= 0.0) return HQ1H;

            double[] Sigma = BuildSEKernel(Nu, dt);

            // γ̃ = γ / trace(HQ1H)
            double trace = 0.0;
            for (int i = 0; i < Nu; i++) trace += HQ1H[i * Nu + i];
            if (trace <= 1e-12) return HQ1H;     // degenerate — skip
            double gammaTilde = gammaRisk / trace;

            // M = HQ1H · Σ  (Nu×Nu)
            double[] M = MatrixOps.MM(HQ1H, Nu, Nu, Sigma, Nu, Nu);
            // M *= 4·γ̃
            for (int i = 0; i < M.Length; i++) M[i] *= 4.0 * gammaTilde;
            // Add I
            for (int i = 0; i < Nu; i++) M[i * Nu + i] += 1.0;

            // HQ1H_risk = M · HQ1H
            return MatrixOps.MM(M, Nu, Nu, HQ1H, Nu, Nu);
        }
    }
}
