// Abstract base for Nash DMPC IBR solvers (longitudinal and lateral).
// Handles prediction-matrix and cost-matrix construction; subclasses
// own the OSQP solver instances and implement the IBR loop.
//
// Prediction matrices (Li et al. 2019, DMPC):
//   U : (Np·nz × nx)  — free-response propagation:  z_free = U · x0
//   H : (Np·nz × Nu)  — input-to-output Toeplitz:   z = H·u + z_free
//
// Cost matrix:
//   Q1_full  — block-diagonal (Np·nz × Np·nz), terminal block scaled up
//   HtQ1     — H' · Q1_full  (Nu × Np·nz), pre-computed for gradient updates
//   HQ1H     — H' · Q1_full · H + reg·I  (Nu × Nu), used in P and q builds
//
// Lambda / alpha mapping (Pustilnik & Borrelli 2025):
//   alpha(lambda) = 1 / (1 + lambda)
//   LongLambdaLevels — 11 levels [0.1 … 50.0]  (mirrors LONG_NASH_LAMBDA_LEVELS in unified/config.py)
//   LatLambdaLevels  —  7 levels [0.1 … 10.0]  (mirrors LAT_NASH_LAMBDA_LEVELS  in unified/config.py)
//   Subclasses select the right set by overriding the virtual LambdaLevels property.
using System;

namespace NashPlatoon
{
    public abstract class NashSolverBase : IDisposable
    {
        // ── Config ────────────────────────────────────────────────────────────
        protected readonly NashWeightsConfig W;
        protected readonly SimTimingConfig   T;

        // ── State-space (discrete-time, set by subclass) ──────────────────────
        protected double[] A;       // nx × nx
        protected double[] B;       // nx × 1
        protected double[] C;       // nz × nx
        protected int Nx, Nz;       // state and output dimensions

        // ── Horizon ───────────────────────────────────────────────────────────
        protected int Np, Nu;

        // ── Prediction matrices ───────────────────────────────────────────────
        protected double[] H;       // Np·nz × Nu
        protected double[] Ht;      // Nu × Np·nz
        protected double[] Umat;    // Np·nz × Nx

        // ── Cost matrices ─────────────────────────────────────────────────────
        protected double[] Q1;      // Np·nz × Np·nz  (block-diagonal)
        protected double[] HtQ1;    // Nu × Np·nz
        protected double[] HQ1H;    // Nu × Nu   (= H'Q1H + reg·I)

        // ── Lambda levels ─────────────────────────────────────────────────────
        // Two canonical sets — subclass selects via the virtual property below.
        protected static readonly double[] LongLambdaLevels =
            { 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 40.0, 50.0 };  // 11 levels — LONG_NASH_LAMBDA_LEVELS
        protected static readonly double[] LatLambdaLevels =
            { 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0 };                           //  7 levels — LAT_NASH_LAMBDA_LEVELS

        // Override in the lateral subclass to return LatLambdaLevels.
        protected virtual double[] LambdaLevels => LongLambdaLevels;

        protected NashSolverBase(NashWeightsConfig w, SimTimingConfig t)
        {
            W  = w;
            T  = t;
            Np = t.NashNp;
            Nu = t.NashNu;
        }

        // ── Prediction matrices ───────────────────────────────────────────────

        protected void RebuildPredictionMatrices()
        {
            // Powers A^0 … A^Np
            var APow = new double[Np + 1][];
            APow[0] = MatrixOps.Identity(Nx);
            for (int k = 1; k <= Np; k++)
                APow[k] = MatrixOps.MM(APow[k - 1], Nx, Nx, A, Nx, Nx);

            // Umat[i] = C · A^(i+1)   (Nz × Nx block at row i·Nz)
            Umat = new double[Np * Nz * Nx];
            for (int i = 0; i < Np; i++)
            {
                double[] block = MatrixOps.MM(C, Nz, Nx, APow[i + 1], Nx, Nx);
                for (int r = 0; r < Nz; r++)
                    for (int c = 0; c < Nx; c++)
                        Umat[(i * Nz + r) * Nx + c] = block[r * Nx + c];
            }

            // CAB[k] = C · A^k · B   (length-Nz vector, B treated as Nx×1)
            var CAB = new double[Np][];
            for (int k = 0; k < Np; k++)
                CAB[k] = MatrixOps.MV(
                    MatrixOps.MM(C, Nz, Nx, APow[k], Nx, Nx), Nz, Nx, B);

            // H[i,j] = CAB[i-j]  for i >= j, else 0  (lower-triangular Toeplitz)
            H = new double[Np * Nz * Nu];
            for (int i = 0; i < Np; i++)
                for (int j = 0; j < Math.Min(i + 1, Nu); j++)
                {
                    double[] cab = CAB[i - j];
                    for (int r = 0; r < Nz; r++)
                        H[(i * Nz + r) * Nu + j] = cab[r];
                }

            Ht = MatrixOps.Transpose(H, Np * Nz, Nu);
        }

        // ── Cost matrices ─────────────────────────────────────────────────────

        protected void RebuildCostMatrices()
        {
            int rows = Np * Nz;

            // Q1_full: block-diagonal, last block gets terminal weights
            Q1 = new double[rows * rows];
            for (int i = 0; i < Np; i++)
            {
                bool terminal = (i == Np - 1);
                // Subclass provides 2D weights via helper — only long case here:
                FillQ1Block(Q1, rows, i, terminal);
            }

            // HtQ1 = Ht · Q1  (Nu × rows)
            HtQ1 = MatrixOps.MM(Ht, Nu, rows, Q1, rows, rows);

            // HQ1H = HtQ1 · H + reg·I  (Nu × Nu)
            HQ1H = MatrixOps.MM(HtQ1, Nu, rows, H, rows, Nu);
            MatrixOps.AddDiagInPlace(HQ1H, Nu, 1e-5);
        }

        // Subclasses override to fill the i-th block of Q1_full appropriately.
        protected virtual void FillQ1Block(double[] Q1full, int rows, int i, bool terminal)
        {
            // Default: 2-state longitudinal  [Q_pos, Q_vel]
            double qp = terminal ? W.Q_pos_terminal : W.Q_pos;
            double qv = terminal ? W.Q_vel_terminal : W.Q_vel;
            Q1full[(i * Nz + 0) * rows + (i * Nz + 0)] = qp;
            Q1full[(i * Nz + 1) * rows + (i * Nz + 1)] = qv;
        }

        // ── Lambda helpers ────────────────────────────────────────────────────

        public static double PustilnikAlpha(double lambda) => 1.0 / (1.0 + lambda);

        /// Nearest-neighbour in log-space against the subclass's LambdaLevels.
        protected int FindLambdaIdx(double lambda)
        {
            var    lev     = LambdaLevels;
            double clamped = Math.Max(lev[0], Math.Min(lev[lev.Length - 1], lambda));
            double logL    = Math.Log(clamped);
            int    best    = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < lev.Length; i++)
            {
                double d = Math.Abs(Math.Log(lev[i]) - logL);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            return best;
        }

        // ── Interface ─────────────────────────────────────────────────────────

        /// Reinitialise all OSQP solvers from updated (Ad, Bd) matrices.
        public abstract void SetupSolvers(double[] Ad, double[] Bd);

        /// Run the IBR loop and return (u1, u2) first-step optimal controls.
        /// x0: state [nx], r1/r2: flat reference [Np·nz], lambda: authority ratio.
        public abstract (float u1, float u2) SolveNashEquilibrium(
            float[] x0, float[] r1, float[] r2, float lambda);

        public abstract void Dispose();
    }
}