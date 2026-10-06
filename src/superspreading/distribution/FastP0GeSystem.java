/*
 * Copyright (C) 2026
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import bdmmprime.distribution.P0GeState;
import bdmmprime.distribution.P0GeSystem;
import bdmmprime.distribution.ScaledNumbers;
import bdmmprime.distribution.SmallNumber;
import bdmmprime.parameterization.Parameterization;
import bdmmprime.util.Utils;

/**
 * {@link P0GeSystem} with faster derivatives and an optional lean integrator
 * (see {@link FastP0System}). The fast integrator must not be enabled when
 * integration results are stored, since it does not produce a continuous
 * output model.
 */
public class FastP0GeSystem extends P0GeSystem {

    protected boolean useFastIntegrator = false;
    private DormandPrince54 fastGeIntegrator;

    protected final double[][] crossRowSum;
    protected final boolean[] hasMigration;

    public FastP0GeSystem(Parameterization parameterization,
                          double absoluteTolerance,
                          double relativeTolerance) {
        super(parameterization, absoluteTolerance, relativeTolerance);
        crossRowSum = FastP0System.crossRowSums(b_ij, M, nTypes, nIntervals);
        hasMigration = FastP0System.hasMigration(M, nTypes, nIntervals);
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
            double pi = y[i], gi = y[nTypes + i];
            double[] bijRow = bijI[i];

            // sum_{j != i} b_ij p_j and sum_{j != i} b_ij g_j
            double crossBirthP = 0.0, crossBirthG = 0.0;
            for (int j = 0; j < nTypes; j++) {
                if (j != i) {
                    crossBirthP += bijRow[j] * y[j];
                    crossBirthG += bijRow[j] * y[nTypes + j];
                }
            }

            double linear = bI[i] + dI[i] + sI[i] + rowSum[i];

            // p0 equations (0 .. nTypes-1)
            yDot[i] = (linear - bI[i] * pi - crossBirthP) * pi - dI[i];

            // ge equations (nTypes .. 2*nTypes-1)
            yDot[nTypes + i] = (linear - 2 * bI[i] * pi - crossBirthP) * gi - pi * crossBirthG;

            if (migration) {
                double[] mRow = mI[i];
                for (int j = 0; j < nTypes; j++) {
                    if (j != i) {
                        yDot[i] -= mRow[j] * y[j];
                        yDot[nTypes + i] -= mRow[j] * y[nTypes + j];
                    }
                }
            }
        }
    }

    /**
     * Same as {@link P0GeSystem#safeIntegrate}, using the lean integrator when
     * it is enabled.
     */
    @Override
    public ScaledNumbers safeIntegrate(ScaledNumbers pgScaled, double tStart, double tEnd) {
        if (!useFastIntegrator)
            return super.safeIntegrate(pgScaled, tStart, tEnd);

        // if the integration interval is too small, nothing is done (to prevent infinite looping)
        if (Utils.equalWithPrecision(tEnd, tStart))
            return pgScaled;

        if (totalProcessLength > 0 && Math.abs(tEnd - tStart) > totalProcessLength / 6) {
            pgScaled = safeIntegrate(pgScaled, tStart, tEnd + (tStart - tEnd) / 2);
            pgScaled = safeIntegrate(pgScaled, tEnd + (tStart - tEnd) / 2, tEnd);
            return pgScaled;
        }

        // Tolerances as in P0GeSystem#safeIntegrate.
        double relativeToleranceConstant = 1e-7;
        double absoluteToleranceConstant = 1e-100;
        int n = pgScaled.getEquation().length / 2;
        double[] absoluteToleranceVector = new double[2 * n];
        double[] relativeToleranceVector = new double[2 * n];
        for (int i = 0; i < n; i++) {
            absoluteToleranceVector[i] = absoluteToleranceConstant;
            if (pgScaled.getEquation()[i + n] > 0)
                absoluteToleranceVector[i + n] = Math.max(1e-310, pgScaled.getEquation()[i + n] * absoluteToleranceConstant);
            else
                absoluteToleranceVector[i + n] = absoluteToleranceConstant;
            relativeToleranceVector[i] = relativeToleranceConstant;
            relativeToleranceVector[i + n] = relativeToleranceConstant;
        }

        if (fastGeIntegrator == null)
            fastGeIntegrator = new DormandPrince54(2 * nTypes, integrationMinStep, integrationMaxStep);

        double[] integrationResults = pgScaled.getEquation().clone();
        int a = pgScaled.getScalingFactor();
        fastGeIntegrator.integrate(this, tStart, integrationResults, tEnd,
                absoluteToleranceVector, relativeToleranceVector);

        double[] pConditions = new double[n];
        SmallNumber[] geConditions = new SmallNumber[n];
        for (int i = 0; i < n; i++) {
            pConditions[i] = integrationResults[i];
            geConditions[i] = new SmallNumber(integrationResults[i + n]);
        }
        pgScaled = (new P0GeState(pConditions, geConditions)).getScaledState();
        pgScaled.augmentFactor(a);
        return pgScaled;
    }
}
