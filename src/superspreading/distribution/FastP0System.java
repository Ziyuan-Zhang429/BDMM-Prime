/*
 * Copyright (C) 2026
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import bdmmprime.distribution.P0State;
import bdmmprime.distribution.P0System;
import bdmmprime.parameterization.Parameterization;

/**
 * {@link P0System} with faster derivatives and an optional lean integrator.
 * <p>
 * The derivatives are algebraically identical to BDMM-Prime's, but the
 * cross-birth row sums are precomputed per interval and the migration terms
 * are skipped when an interval has no migration. With
 * {@link #setUseFastIntegrator(boolean)} the integration uses
 * {@link DormandPrince54} (same method as the commons-math integrator,
 * without its per-call overhead).
 */
public class FastP0System extends P0System {

    protected final double absoluteTolerance, relativeTolerance;

    protected boolean useFastIntegrator = false;
    private DormandPrince54 fastP0Integrator;
    private double[] p0AbsTol, p0RelTol;

    // Per interval and type: sum over j != i of (b_ij + M_ij), and whether
    // the interval has any migration.
    protected final double[][] crossRowSum;
    protected final boolean[] hasMigration;

    public FastP0System(Parameterization parameterization,
                        double absoluteTolerance,
                        double relativeTolerance) {
        super(parameterization, absoluteTolerance, relativeTolerance);
        this.absoluteTolerance = absoluteTolerance;
        this.relativeTolerance = relativeTolerance;
        crossRowSum = crossRowSums(b_ij, M, nTypes, nIntervals);
        hasMigration = hasMigration(M, nTypes, nIntervals);
    }

    /**
     * Use the lean Dormand-Prince integrator instead of the commons-math one.
     */
    public void setUseFastIntegrator(boolean useFastIntegrator) {
        this.useFastIntegrator = useFastIntegrator;
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) {

        double[] bI = b[interval], dI = d[interval], sI = s[interval], rowSum = crossRowSum[interval];
        double[][] bijI = b_ij[interval], mI = M[interval];
        boolean migration = hasMigration[interval];

        for (int i = 0; i < nTypes; i++) {
            double yi = y[i];
            double[] bijRow = bijI[i];

            // sum_{j != i} b_ij y_j
            double crossBirth = 0.0;
            for (int j = 0; j < nTypes; j++)
                if (j != i)
                    crossBirth += bijRow[j] * y[j];

            yDot[i] = (bI[i] + dI[i] + sI[i] + rowSum[i] - bI[i] * yi - crossBirth) * yi - dI[i];

            if (migration) {
                double[] mRow = mI[i];
                for (int j = 0; j < nTypes; j++)
                    if (j != i)
                        yDot[i] -= mRow[j] * y[j];
            }
        }
    }

    @Override
    public void integrate(P0State state, double tStart, double tEnd) {
        if (!useFastIntegrator) {
            super.integrate(state, tStart, tEnd);
            return;
        }

        if (fastP0Integrator == null) {
            fastP0Integrator = new DormandPrince54(nTypes, integrationMinStep, integrationMaxStep);
            p0AbsTol = new double[nTypes];
            p0RelTol = new double[nTypes];
            java.util.Arrays.fill(p0AbsTol, absoluteTolerance);
            java.util.Arrays.fill(p0RelTol, relativeTolerance);
        }
        fastP0Integrator.integrate(this, tStart, state.p0, tEnd, p0AbsTol, p0RelTol);
    }

    static double[][] crossRowSums(double[][][] b_ij, double[][][] M, int nTypes, int nIntervals) {
        double[][] sums = new double[nIntervals][nTypes];
        for (int interval = 0; interval < nIntervals; interval++)
            for (int i = 0; i < nTypes; i++)
                for (int j = 0; j < nTypes; j++)
                    if (i != j)
                        sums[interval][i] += b_ij[interval][i][j] + M[interval][i][j];
        return sums;
    }

    static boolean[] hasMigration(double[][][] M, int nTypes, int nIntervals) {
        boolean[] result = new boolean[nIntervals];
        for (int interval = 0; interval < nIntervals; interval++)
            for (int i = 0; i < nTypes; i++)
                for (int j = 0; j < nTypes; j++)
                    if (i != j && M[interval][i][j] != 0.0)
                        result[interval] = true;
        return result;
    }
}
