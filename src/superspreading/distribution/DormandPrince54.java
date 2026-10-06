/*
 * Copyright (C) 2026
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;

/**
 * Lean Dormand-Prince 5(4) integrator.
 * <p>
 * Same method as commons-math's DormandPrince54Integrator (Butcher tableau,
 * embedded error estimate, error norm, step size control and initial step
 * heuristic), but without step handlers, event handling, state copies or
 * per-call allocations: all work arrays are allocated once and reused. Not
 * thread safe; use one instance per thread.
 */
public final class DormandPrince54 {

    private static final double[] C = {0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0};

    private static final double[][] A = {
            {},
            {1.0 / 5.0},
            {3.0 / 40.0, 9.0 / 40.0},
            {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0},
            {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0},
            {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0},
            {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0}};

    // Error estimation coefficients (difference between 5th and 4th order solutions).
    private static final double E1 = 71.0 / 57600.0, E3 = -71.0 / 16695.0, E4 = 71.0 / 1920.0,
            E5 = -17253.0 / 339200.0, E6 = 22.0 / 525.0, E7 = -1.0 / 40.0;

    private static final double SAFETY = 0.9, MIN_REDUCTION = 0.2, MAX_GROWTH = 10.0;
    private static final double EXPONENT = -1.0 / 5.0;

    private final int n;
    private final double minStep, maxStep;
    private final double[][] k;
    private final double[] yTmp, scale;

    public DormandPrince54(int dimension, double minStep, double maxStep) {
        this.n = dimension;
        this.minStep = Math.abs(minStep);
        this.maxStep = Math.abs(maxStep);
        this.k = new double[7][dimension];
        this.yTmp = new double[dimension];
        this.scale = new double[dimension];
    }

    /**
     * Integrates the equations from t0 to t, updating y in place.
     *
     * @param absTol absolute tolerance per component
     * @param relTol relative tolerance per component
     */
    public void integrate(FirstOrderDifferentialEquations equations, double t0, double[] y, double t,
                          double[] absTol, double[] relTol) {
        if (t0 == t)
            return;

        final boolean forward = t > t0;
        double stepStart = t0;

        equations.computeDerivatives(t0, y, k[0]);

        for (int i = 0; i < n; i++)
            scale[i] = absTol[i] + relTol[i] * Math.abs(y[i]);
        double hNew = initializeStep(equations, forward, t0, y, absTol, relTol);

        while (true) {
            double stepSize;
            double error;
            boolean lastStep;
            do {
                stepSize = hNew;
                lastStep = forward ? (stepStart + stepSize >= t) : (stepStart + stepSize <= t);
                if (lastStep)
                    stepSize = t - stepStart;

                // stages 2 to 7 (stage 7 is evaluated at the end-of-step state, FSAL)
                for (int s = 1; s < 7; s++) {
                    for (int i = 0; i < n; i++) {
                        double sum = A[s][0] * k[0][i];
                        for (int l = 1; l < s; l++)
                            sum += A[s][l] * k[l][i];
                        yTmp[i] = y[i] + stepSize * sum;
                    }
                    equations.computeDerivatives(stepStart + C[s] * stepSize, yTmp, k[s]);
                }
                // yTmp now holds the 5th order solution at the end of the step.

                error = estimateError(y, yTmp, stepSize, absTol, relTol);
                if (error >= 1.0) {
                    double factor = Math.min(MAX_GROWTH,
                            Math.max(MIN_REDUCTION, SAFETY * Math.pow(error, EXPONENT)));
                    hNew = filterStep(stepSize * factor, forward, false);
                }
            } while (error >= 1.0);

            // accept the step
            System.arraycopy(yTmp, 0, y, 0, n);
            if (lastStep)
                return;
            stepStart += stepSize;
            System.arraycopy(k[6], 0, k[0], 0, n);

            double factor = Math.min(MAX_GROWTH,
                    Math.max(MIN_REDUCTION, SAFETY * Math.pow(error, EXPONENT)));
            double scaledH = stepSize * factor;
            boolean nextIsLast = forward ? (stepStart + scaledH >= t) : (stepStart + scaledH <= t);
            hNew = filterStep(scaledH, forward, nextIsLast);
        }
    }

    private double estimateError(double[] y0, double[] y1, double h, double[] absTol, double[] relTol) {
        double error = 0;
        for (int j = 0; j < n; j++) {
            double errSum = E1 * k[0][j] + E3 * k[2][j] + E4 * k[3][j]
                    + E5 * k[4][j] + E6 * k[5][j] + E7 * k[6][j];
            double yScale = Math.max(Math.abs(y0[j]), Math.abs(y1[j]));
            double tol = absTol[j] + relTol[j] * yScale;
            double ratio = h * errSum / tol;
            error += ratio * ratio;
        }
        return Math.sqrt(error / n);
    }

    /**
     * Initial step size heuristic (Hairer, Norsett and Wanner), as in
     * commons-math's AdaptiveStepsizeIntegrator. Uses k[0] (derivative at t0)
     * and k[1] / yTmp as scratch space.
     */
    private double initializeStep(FirstOrderDifferentialEquations equations, boolean forward,
                                  double t0, double[] y0, double[] absTol, double[] relTol) {
        double yOnScale2 = 0, yDotOnScale2 = 0;
        for (int j = 0; j < n; j++) {
            double ratio = y0[j] / scale[j];
            yOnScale2 += ratio * ratio;
            ratio = k[0][j] / scale[j];
            yDotOnScale2 += ratio * ratio;
        }

        double h = (yOnScale2 < 1.0e-10 || yDotOnScale2 < 1.0e-10)
                ? 1.0e-6 : 0.01 * Math.sqrt(yOnScale2 / yDotOnScale2);
        if (!forward)
            h = -h;

        for (int j = 0; j < n; j++)
            yTmp[j] = y0[j] + h * k[0][j];
        equations.computeDerivatives(t0 + h, yTmp, k[1]);

        double yDDotOnScale = 0;
        for (int j = 0; j < n; j++) {
            double ratio = (k[1][j] - k[0][j]) / scale[j];
            yDDotOnScale += ratio * ratio;
        }
        yDDotOnScale = Math.sqrt(yDDotOnScale) / h;

        double maxInv2 = Math.max(Math.sqrt(yDotOnScale2), yDDotOnScale);
        double h1 = (maxInv2 < 1.0e-15)
                ? Math.max(1.0e-6, 0.001 * Math.abs(h))
                : Math.pow(0.01 / maxInv2, 1.0 / 5.0);
        h = Math.min(100.0 * Math.abs(h), h1);
        h = Math.max(h, 1.0e-12 * Math.abs(t0));
        if (h < minStep)
            h = minStep;
        if (h > maxStep)
            h = maxStep;

        return forward ? h : -h;
    }

    private double filterStep(double h, boolean forward, boolean acceptSmall) {
        double filtered = h;
        if (Math.abs(h) < minStep) {
            if (acceptSmall)
                filtered = forward ? minStep : -minStep;
            else
                throw new NumberIsTooSmallException(LocalizedFormats.MINIMAL_STEPSIZE_REACHED_DURING_INTEGRATION,
                        Math.abs(h), minStep, true);
        }
        if (filtered > maxStep)
            filtered = maxStep;
        else if (filtered < -maxStep)
            filtered = -maxStep;
        return filtered;
    }
}
