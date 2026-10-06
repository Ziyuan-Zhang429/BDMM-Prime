/*
 * Copyright (C) 2025
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import bdmmprime.parameterization.Parameterization;

/**
 * Mean-field version of {@link P0System} for parameterizations whose cross-birth
 * matrix is rank-1 and which have no migration between lineages, i.e.
 *
 *   b_ij[i][j] = c_i * w_j,   b[i] = c_i * w_i,   M[i][j] = 0,
 *
 * where c_i is the total birth rate of a type-i lineage and {w_j} the (shared)
 * offspring category distribution. This holds for
 * {@link bdmmprime.parameterization.NegativeBinomialApproximation}: an offspring
 * draws its category from {w_j} independently of its parent.
 *
 * Under this structure the p0 equation
 *
 *   dp0_i/dt = (b_i+d_i+s_i) p0_i - d_i - b_i p0_i^2 - sum_{j!=i} b_ij p0_i p0_j
 *
 * collapses (the quadratic coupling factorises through a single scalar
 * A = sum_j w_j p0_j) to
 *
 *   dp0_i/dt = (c_i + d_i + s_i) p0_i - d_i - c_i p0_i A.
 *
 * This is mathematically identical to {@link P0System} for a rank-1, no-migration
 * parameterization, but costs O(J) per step instead of O(J^2) and has a
 * diagonal-plus-rank-1 Jacobian.
 */
public class NBP0System extends FastP0System {

    /** c[interval][i]: total birth rate of a type-i lineage. */
    protected final double[][] c;
    /** w[interval][j]: offspring category weights (sum to 1 over j). */
    protected final double[][] w;

    public NBP0System(Parameterization parameterization,
                      double absoluteTolerance,
                      double relativeTolerance) {
        super(parameterization, absoluteTolerance, relativeTolerance);
        requireNoMigration(M, nTypes, nIntervals);
        c = totalBirthRates(b, b_ij, nTypes, nIntervals);
        w = categoryWeights(b, b_ij, c, nTypes, nIntervals);
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) {
        double A = 0.0;
        for (int j = 0; j < nTypes; j++)
            A += w[interval][j] * y[j];

        for (int i = 0; i < nTypes; i++) {
            double ci = c[interval][i];
            yDot[i] = (ci + d[interval][i] + s[interval][i]) * y[i]
                    - d[interval][i]
                    - ci * y[i] * A;
        }
    }

    // ===== shared rank-1 factorisation utilities (used by NBP0GeSystem too) =====

    /** Throw if any migration rate is non-zero (mean-field form assumes none). */
    static void requireNoMigration(double[][][] M, int nTypes, int nIntervals) {
        for (int iv = 0; iv < nIntervals; iv++)
            for (int i = 0; i < nTypes; i++)
                for (int j = 0; j < nTypes; j++)
                    if (M[iv][i][j] != 0.0)
                        throw new IllegalArgumentException(
                                "Mean-field ODE system requires zero migration rates.");
    }

    /** c[interval][i] = b[i] + sum_{j!=i} b_ij[i][j] (total birth rate of type i). */
    static double[][] totalBirthRates(double[][] b, double[][][] b_ij,
                                      int nTypes, int nIntervals) {
        double[][] c = new double[nIntervals][nTypes];
        for (int iv = 0; iv < nIntervals; iv++) {
            for (int i = 0; i < nTypes; i++) {
                double tot = b[iv][i];
                for (int j = 0; j < nTypes; j++)
                    if (j != i)
                        tot += b_ij[iv][i][j];
                c[iv][i] = tot;
            }
        }
        return c;
    }

    /**
     * Extract the shared offspring weights w_j from the rank-1 cross-birth
     * matrix, using the row with the largest total birth rate for numerical
     * stability, and verify the rank-1 structure b_ij[i][j] == c_i * w_j.
     */
    static double[][] categoryWeights(double[][] b, double[][][] b_ij,
                                      double[][] c, int nTypes, int nIntervals) {
        double[][] w = new double[nIntervals][nTypes];
        for (int iv = 0; iv < nIntervals; iv++) {

            // reference row = argmax c (most transmissible, best conditioned)
            int ref = 0;
            for (int i = 1; i < nTypes; i++)
                if (c[iv][i] > c[iv][ref]) ref = i;

            if (c[iv][ref] <= 0.0) {
                // no births at all in this interval: weights are irrelevant
                // (they are always multiplied by c_i = 0). Use uniform.
                for (int j = 0; j < nTypes; j++) w[iv][j] = 1.0 / nTypes;
                continue;
            }

            for (int j = 0; j < nTypes; j++)
                w[iv][j] = (j == ref ? b[iv][ref] : b_ij[iv][ref][j]) / c[iv][ref];

            // verify rank-1: b_ij[i][j] == c_i * w_j and b[i] == c_i * w_i
            for (int i = 0; i < nTypes; i++) {
                for (int j = 0; j < nTypes; j++) {
                    double expected = c[iv][i] * w[iv][j];
                    double actual = (j == i) ? b[iv][i] : b_ij[iv][i][j];
                    double scale = Math.max(1e-100, Math.abs(actual) + Math.abs(expected));
                    if (Math.abs(actual - expected) > 1e-6 * scale)
                        throw new IllegalArgumentException(
                                "Mean-field ODE system requires a rank-1 cross-birth "
                                + "matrix (b_ij[i][j] = c_i * w_j); parameterization is "
                                + "not compatible (interval " + iv + ", i=" + i + ", j=" + j + ").");
                }
            }
        }
        return w;
    }
}
