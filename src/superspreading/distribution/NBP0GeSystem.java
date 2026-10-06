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
 * Mean-field version of {@link P0GeSystem} for rank-1, no-migration
 * parameterizations (see {@link NBP0System}). With
 * b_ij[i][j] = c_i w_j, b[i] = c_i w_i and M = 0, both the p0 and the ge
 * equations factorise through shared scalars
 *
 *   A = sum_j w_j p0_j,   G = sum_j w_j ge_j,
 *
 * giving
 *
 *   dp0_i/dt = (c_i + d_i + s_i) p0_i - d_i - c_i p0_i A
 *   dge_i/dt = (c_i + d_i + s_i) ge_i - c_i (A ge_i + G p0_i)
 *
 * (the w_i p0_i ge_i cross-terms cancel exactly). This is mathematically
 * identical to {@link P0GeSystem} for such parameterizations, but O(J) per step.
 */
public class NBP0GeSystem extends FastP0GeSystem {

    protected final double[][] c;
    protected final double[][] w;

    public NBP0GeSystem(Parameterization parameterization,
                        double absoluteTolerance,
                        double relativeTolerance) {
        super(parameterization, absoluteTolerance, relativeTolerance);
        NBP0System.requireNoMigration(M, nTypes, nIntervals);
        c = NBP0System.totalBirthRates(b, b_ij, nTypes, nIntervals);
        w = NBP0System.categoryWeights(b, b_ij, c, nTypes, nIntervals);
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) {
        double A = 0.0, G = 0.0;
        for (int j = 0; j < nTypes; j++) {
            A += w[interval][j] * y[j];
            G += w[interval][j] * y[nTypes + j];
        }

        for (int i = 0; i < nTypes; i++) {
            double ci = c[interval][i];
            double linCoef = ci + d[interval][i] + s[interval][i];

            // p0 equations (0 .. nTypes-1)
            yDot[i] = linCoef * y[i] - d[interval][i] - ci * y[i] * A;

            // ge equations (nTypes .. 2*nTypes-1)
            yDot[nTypes + i] = linCoef * y[nTypes + i]
                    - ci * (A * y[nTypes + i] + G * y[i]);
        }
    }
}
