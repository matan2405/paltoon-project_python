// Row-major flat-array matrix utilities for the Nash DMPC solver.
// Convention: M[r*cols + c].  All methods are allocation-minimising but
// allocate a new array for the result unless the name ends in "InPlace".
using System;

namespace NashPlatoon
{
    internal static class MatrixOps
    {
        // ── Basic algebra ─────────────────────────────────────────────────────

        /// A(rA×cA) × B(rB×cB) → C(rA×cB);  cA must equal rB.
        public static double[] MM(double[] A, int rA, int cA, double[] B, int rB, int cB)
        {
            var C = new double[rA * cB];
            for (int i = 0; i < rA; i++)
                for (int k = 0; k < cA; k++)
                {
                    double a = A[i * cA + k];
                    for (int j = 0; j < cB; j++)
                        C[i * cB + j] += a * B[k * cB + j];
                }
            return C;
        }

        /// M(rows×cols) × x(cols) → y(rows).
        public static double[] MV(double[] M, int rows, int cols, double[] x)
        {
            var y = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                double s = 0;
                for (int j = 0; j < cols; j++) s += M[i * cols + j] * x[j];
                y[i] = s;
            }
            return y;
        }

        /// M(rows×cols) transposed → M'(cols×rows).
        public static double[] Transpose(double[] M, int rows, int cols)
        {
            var T = new double[cols * rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    T[j * rows + i] = M[i * cols + j];
            return T;
        }

        public static double[] Add(double[] a, double[] b)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = a[i] + b[i];
            return c;
        }

        public static double[] Sub(double[] a, double[] b)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = a[i] - b[i];
            return c;
        }

        public static double[] Scale(double[] a, double s)
        {
            var c = new double[a.Length];
            for (int i = 0; i < a.Length; i++) c[i] = s * a[i];
            return c;
        }

        /// a += s * b  (in-place, no allocation).
        public static void AddScaleInPlace(double[] a, double[] b, double s)
        {
            for (int i = 0; i < a.Length; i++) a[i] += s * b[i];
        }

        public static double MaxAbs(double[] a)
        {
            double m = 0;
            for (int i = 0; i < a.Length; i++) { double v = Math.Abs(a[i]); if (v > m) m = v; }
            return m;
        }

        public static double MaxAbs2(double[] a, double[] b) => Math.Max(MaxAbs(a), MaxAbs(b));

        public static double Dot(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }

        // ── Matrix construction ───────────────────────────────────────────────

        public static double[] Identity(int n)
        {
            var I = new double[n * n];
            for (int i = 0; i < n; i++) I[i * n + i] = 1.0;
            return I;
        }

        /// A[i,i] += scale  for all i (square n×n in-place).
        public static void AddDiagInPlace(double[] A, int n, double scale)
        {
            for (int i = 0; i < n; i++) A[i * n + i] += scale;
        }

        /// A = 0.5*(A + A')  in-place (enforces exact symmetry).
        public static void SymmetrizeInPlace(double[] A, int n)
        {
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double s = 0.5 * (A[i * n + j] + A[j * n + i]);
                    A[i * n + j] = s;
                    A[j * n + i] = s;
                }
        }

        public static double[] Copy(double[] a)
        {
            var b = new double[a.Length];
            Array.Copy(a, b, a.Length);
            return b;
        }

        /// Shift left by 1, repeating the last element (MPC horizon shift for warm-start).
        public static double[] ShiftLeft(double[] a)
        {
            var b = new double[a.Length];
            for (int i = 0; i < a.Length - 1; i++) b[i] = a[i + 1];
            b[a.Length - 1] = a[a.Length - 1];
            return b;
        }

        // ── Matrix exponential ────────────────────────────────────────────────

        /// Matrix exponential via Padé [3/3] with scaling and squaring.
        /// Sufficient for ZOH augmented matrices whose 1-norm is typically << 1.
        public static double[] Expm(double[] M, int n)
        {
            double norm = OneNorm(M, n);
            int s = 0;
            double[] Ms = Copy(M);
            while (norm > 0.5)
            {
                norm *= 0.5; s++;
                for (int i = 0; i < Ms.Length; i++) Ms[i] *= 0.5;
            }

            // Padé [3/3] coefficients
            const double c1 = 0.5, c2 = 0.1, c3 = 1.0 / 120.0;
            double[] I3  = Identity(n);
            double[] A2  = MM(Ms, n, n, Ms, n, n);
            double[] A3  = MM(A2, n, n, Ms, n, n);

            double[] p = Copy(I3);
            AddScaleInPlace(p, Ms,  c1);
            AddScaleInPlace(p, A2,  c2);
            AddScaleInPlace(p, A3,  c3);

            double[] q = Copy(I3);
            AddScaleInPlace(q, Ms, -c1);
            AddScaleInPlace(q, A2,  c2);
            AddScaleInPlace(q, A3, -c3);

            double[] R = SolveMatrixEquation(q, n, p);  // expm(Ms) = q^{-1} p

            for (int i = 0; i < s; i++)
                R = MM(R, n, n, R, n, n);

            return R;
        }

        /// ZOH discretisation: given continuous-time (Ac n×n, Bc n×nu) and dt,
        /// returns (Ad n×n, Bd n×nu) via matrix exponential of the augmented system.
        public static (double[] Ad, double[] Bd) ZOH(double[] Ac, double[] Bc,
                                                      int n, int nu, double dt)
        {
            int aug = n + nu;
            var M = new double[aug * aug];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n;  j++) M[i * aug + j]       = Ac[i * n  + j] * dt;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < nu; j++) M[i * aug + (n + j)] = Bc[i * nu + j] * dt;

            double[] eM = Expm(M, aug);

            double[] Ad = new double[n * n];
            double[] Bd = new double[n * nu];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n;  j++) Ad[i * n  + j] = eM[i * aug + j];
                for (int j = 0; j < nu; j++) Bd[i * nu + j] = eM[i * aug + (n + j)];
            }
            return (Ad, Bd);
        }

        // ── CSC conversion (for OSQP) ─────────────────────────────────────────

        /// Dense symmetric n×n → upper-triangular CSC (OSQP P format).
        /// All upper-triangle entries (including structural zeros) are stored
        /// to keep a uniform sparsity pattern independent of numerical values.
        public static (double[] data, int[] indices, int[] indptr)
            DenseSymToUpperCsc(double[] M, int n)
        {
            int nnz = n * (n + 1) / 2;
            var data    = new double[nnz];
            var indices = new int[nnz];
            var indptr  = new int[n + 1];

            int p = 0;
            for (int j = 0; j < n; j++)
            {
                indptr[j] = p;
                for (int i = 0; i <= j; i++) { data[p] = M[i * n + j]; indices[p] = i; p++; }
            }
            indptr[n] = p;
            return (data, indices, indptr);
        }

        /// Dense (rows × cols) → general CSC (OSQP A format).
        /// Skips exact zeros to keep the constraint matrix sparse.
        public static (double[] data, int[] indices, int[] indptr)
            DenseToCSC(double[] M, int rows, int cols)
        {
            int nnz = 0;
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < rows; i++)
                    if (M[i * cols + j] != 0.0) nnz++;

            var data    = new double[nnz];
            var indices = new int[nnz];
            var indptr  = new int[cols + 1];

            int p = 0;
            for (int j = 0; j < cols; j++)
            {
                indptr[j] = p;
                for (int i = 0; i < rows; i++)
                {
                    double v = M[i * cols + j];
                    if (v != 0.0) { data[p] = v; indices[p] = i; p++; }
                }
            }
            indptr[cols] = p;
            return (data, indices, indptr);
        }

        // ── Private helpers ───────────────────────────────────────────────────

        static double OneNorm(double[] M, int n)
        {
            double max = 0;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < n; i++) s += Math.Abs(M[i * n + j]);
                if (s > max) max = s;
            }
            return max;
        }

        // Solve Q·X = P for X via Gauss-Jordan on the augmented [Q | P] (n × 2n).
        static double[] SolveMatrixEquation(double[] Q, int n, double[] P)
        {
            var aug = new double[n * 2 * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) aug[i * 2*n + j]     = Q[i * n + j];
                for (int j = 0; j < n; j++) aug[i * 2*n + n + j] = P[i * n + j];
            }

            for (int col = 0; col < n; col++)
            {
                // Partial pivot
                int    pivot   = col;
                double maxVal  = Math.Abs(aug[col * 2*n + col]);
                for (int row = col + 1; row < n; row++)
                {
                    double v = Math.Abs(aug[row * 2*n + col]);
                    if (v > maxVal) { maxVal = v; pivot = row; }
                }
                if (pivot != col)
                    for (int j = 0; j < 2*n; j++)
                    {
                        double tmp = aug[col * 2*n + j];
                        aug[col * 2*n + j]   = aug[pivot * 2*n + j];
                        aug[pivot * 2*n + j] = tmp;
                    }

                double diag = aug[col * 2*n + col];
                if (Math.Abs(diag) < 1e-14) continue;
                for (int j = col; j < 2*n; j++) aug[col * 2*n + j] /= diag;

                for (int row = 0; row < n; row++)
                {
                    if (row == col) continue;
                    double f = aug[row * 2*n + col];
                    if (f == 0.0) continue;
                    for (int j = col; j < 2*n; j++)
                        aug[row * 2*n + j] -= f * aug[col * 2*n + j];
                }
            }

            var X = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    X[i * n + j] = aug[i * 2*n + n + j];
            return X;
        }
    }
}
