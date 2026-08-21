/*
 * Copyright (C) 2025
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package bdmmprime.parameterization;

import beast.base.core.Input;
import beast.base.core.Loggable;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;

import java.io.PrintStream;

/**
 * Parameterization that approximates negative binomial offspring distribution
 * through discretized Gamma heterogeneity in transmission rates.
 */
public class NegativeBinomialApproximation extends Parameterization implements Loggable {

    // ===== INPUTS =====

    public Input<SkylineVectorParameter> meanRInput = new Input<>(
            "meanR",
            "Mean reproduction number (R)",
            Input.Validate.REQUIRED);

    public Input<SkylineVectorParameter> kInput = new Input<>(
            "k",
            "Dispersion parameter (smaller = more superspreading)",
            Input.Validate.REQUIRED);

    public Input<SkylineVectorParameter> samplingPropInput = new Input<>(
            "samplingProp",
            "Sampling proportion (psi)",
            Input.Validate.REQUIRED);

    public Input<SkylineVectorParameter> removalRateInput = new Input<>(
            "removalRate",
            "Total removal rate (nu)",
            Input.Validate.REQUIRED);

    public Input<TimedParameter> rhoSamplingInput = new Input<>(
            "rhoSampling",
            "Contemporaneous sampling (optional)");

    public Input<Integer> nTypesInput = new Input<>(
            "nTypes",
            "Number of discrete types for Gamma approximation",
            5);  // default value

    public Input<Boolean> gaussLaguerreInput = new Input<>(
            "gaussLaguerre",
            "Discretisation scheme for the Gamma transmissibility distribution. "
            + "false (default): equal-probability bins with equal category weights "
            + "(1/J) and conditional-mean rates. true: generalized Gauss-Laguerre "
            + "quadrature with UNEQUAL category weights, which matches the first "
            + "2J-1 moments of the Gamma exactly (so mean AND variance are captured "
            + "with far fewer categories).",
            false);


    // ===== CALCULATED VALUES =====

    private double[] lambdaValues;
    private double[] categoryWeights;
    private double[] calculatedBirthRates;
    private double[][] calculatedCrossBirthRates;
    private double[] calculatedDeathRates;
    private double[] calculatedSamplingRates;

    // Absolute accuracy for the Gamma inverse-CDF (quantile) solver. The
    // Commons-Math default (1e-9) is far coarser than the true lower-tail
    // quantiles at small k (~1e-13), which mislocates the equal-probability
    // bin edges (producing unequal bin weights). Tightening it makes the
    // edges — and hence the bin weights — correct.
    private static final double INV_CDF_ACCURACY = 1e-20;


    // ===== INITIALIZATION =====

    @Override
    public void initAndValidate() {
        // Set nTypes from input or use default
        int requestedTypes = nTypesInput.get();
        if (requestedTypes < 2) {
            throw new IllegalArgumentException(
                    "NegativeBinomialApproximation requires nTypes >= 2");
        }

        // The discretisation categories ARE the model's types. The base
        // Parameterization derives nTypes from the typeSet and eagerly
        // allocates its rate arrays inside super.initAndValidate(), so the
        // typeSet must already carry `requestedTypes` types BEFORE super runs;
        // otherwise the base is sized for the (scalar) input typeSet and the
        // rate arrays end up the wrong length (ArrayIndexOutOfBounds, or a
        // silently wrong likelihood -> nonsensical R at nTypes > 1).
        typeSetInput.setValue(new TypeSet(requestedTypes), this);

        super.initAndValidate();
        // nTypes == requestedTypes from here on.

        System.out.println("[NegativeBinomialApproximation] initialised "
                + "(numerical-guard build) with nTypes=" + nTypes);
        printDiscretizationDebug(0.0);
    }


    // ===== CHANGE TIMES =====

    @Override
    public double[] getBirthRateChangeTimes() {
        return combineAndSortTimes(null,
                meanRInput.get().getChangeTimes(),
                kInput.get().getChangeTimes(),
                removalRateInput.get().getChangeTimes());
    }

    @Override
    public double[] getDeathRateChangeTimes() {
        return combineAndSortTimes(null,
                removalRateInput.get().getChangeTimes(),
                samplingPropInput.get().getChangeTimes());
    }

    @Override
    public double[] getSamplingRateChangeTimes() {
        return combineAndSortTimes(null,
                removalRateInput.get().getChangeTimes(),
                samplingPropInput.get().getChangeTimes());
    }

    @Override
    public double[] getCrossBirthRateChangeTimes() {
        return getBirthRateChangeTimes();
    }

    @Override
    public double[] getMigRateChangeTimes() {
        return EMPTY_TIME_ARRAY;
    }

    @Override
    public double[] getRemovalProbChangeTimes() {
        return EMPTY_TIME_ARRAY;
    }

    @Override
    public double[] getRhoSamplingTimes() {
        if (rhoSamplingInput.get() != null)
            return rhoSamplingInput.get().getTimes();
        return EMPTY_TIME_ARRAY;
    }


    // ===== RATE VALUES =====

    @Override
    protected double[] getBirthRateValues(double time) {
        recalculateRates(time);
        return calculatedBirthRates;
    }

    @Override
    protected double[][] getCrossBirthRateValues(double time) {
        return calculatedCrossBirthRates;
    }

    @Override
    protected double[] getDeathRateValues(double time) {
        return calculatedDeathRates;
    }

    @Override
    protected double[] getSamplingRateValues(double time) {
        return calculatedSamplingRates;
    }

    @Override
    protected double[][] getMigRateValues(double time) {
        return ZERO_VALUE_MATRIX;
    }

    @Override
    protected double[] getRemovalProbValues(double time) {
        int J = getNTypes();
        double[] ones = new double[J];
        for (int j = 0; j < J; j++) {
            ones[j] = 1.0;
        }
        return ones;
    }

    @Override
    protected double[] getRhoValues(double time) {
        if (rhoSamplingInput.get() != null)
            return rhoSamplingInput.get().getValuesAtTime(time);
        return ZERO_VALUE_ARRAY;
    }


    // ===== CORE CALCULATION =====

    /**
     * The category weights {w_j} (probability that a new infection falls in
     * category j), summing to 1. For the equal-probability scheme these are
     * all 1/J; for Gauss-Laguerre they are the (unequal) quadrature weights.
     * This is the stationary type distribution of the process, and is the
     * correct choice for the BDMM origin type prior (startTypePriorProbs).
     * Evaluated at the start of the process (time 0).
     */
    public double[] getCategoryWeights() {
        recalculateRates(0.0);
        return categoryWeights.clone();
    }

    private void recalculateRates(double time) {
        int J = getNTypes();

        if (lambdaValues == null) {
            lambdaValues = new double[J];
            categoryWeights = new double[J];
            calculatedBirthRates = new double[J];
            calculatedCrossBirthRates = new double[J][J];
            calculatedDeathRates = new double[J];
            calculatedSamplingRates = new double[J];
        }

        double meanR = meanRInput.get().getValuesAtTime(time)[0];
        double k = kInput.get().getValuesAtTime(time)[0];
        double psi = samplingPropInput.get().getValuesAtTime(time)[0];
        double nu = removalRateInput.get().getValuesAtTime(time)[0];

        if (gaussLaguerreInput.get()) {

            // ================= VERSION 2: Gauss-Laguerre =================
            // Generalized Gauss-Laguerre quadrature: UNEQUAL category weights
            // w_j and nodes chosen so the first 2J-1 moments of the Gamma match
            // exactly (mean AND variance captured). An offspring draws its
            // category from {w_k}, so the type-j -> type-k birth rate is
            // nu*lambda_j*w_k. birthRate[j] carries the same-type channel
            // (nu*lambda_j*w_j) and crossBirth[j][k] the cross-type channels;
            // total = nu*lambda_j*sum_k w_k = nu*lambda_j.
            gaussLaguerreCategories(meanR, k, lambdaValues, categoryWeights);

            for (int j = 0; j < J; j++) {
                double base = nu * lambdaValues[j];
                calculatedBirthRates[j] = base * categoryWeights[j];
                for (int kk = 0; kk < J; kk++) {
                    calculatedCrossBirthRates[j][kk] = base * categoryWeights[kk];
                }
            }

        } else {

            // ============ VERSION 1: equal-probability bins ============
            // J equal-probability bins (weight 1/J each), rate = conditional
            // mean of each bin. An offspring draws its type uniformly, so the
            // birth rate to EVERY destination type is nu*lambda_j/J. BDMM's ODE
            // uses birthRate[j] for the same-type channel (j->j) and
            // crossBirth[j][k] for cross-type channels (j->k, k!=j; diagonal
            // ignored). within == cross == nu*lambda_j/J, giving total
            // birthRate[j] + sum_{k!=j} crossBirth[j][k] = nu*lambda_j.
            discretizeGamma(meanR, k, lambdaValues);

            double pDest = 1.0 / J;
            for (int j = 0; j < J; j++) {
                categoryWeights[j] = pDest;
                double perDestBirthRate = nu * lambdaValues[j] * pDest;
                calculatedBirthRates[j] = perDestBirthRate;
                for (int kk = 0; kk < J; kk++) {
                    calculatedCrossBirthRates[j][kk] = perDestBirthRate;
                }
            }
        }

        double samplingRate = nu * psi;
        for (int j = 0; j < J; j++) {
            calculatedSamplingRates[j] = samplingRate;
        }

        double deathRate = nu * (1.0 - psi);
        for (int j = 0; j < J; j++) {
            calculatedDeathRates[j] = deathRate;
        }
    }

    /**
     * VERSION 2 discretisation: generalized Gauss-Laguerre quadrature for
     * Gamma(shape=k, scale=R/k). Fills lambdaOut with the J category rates
     * (lambda_j = (R/k) * node_j) and weightOut with the probability weights
     * (sum to 1), such that sum_j weightOut[j]*lambdaOut[j]^m equals E[lambda^m]
     * for every moment m <= 2J-1 (so mean and variance are matched exactly).
     *
     * Golub-Welsch: the nodes are the eigenvalues of the symmetric tridiagonal
     * Jacobi matrix of the generalized Laguerre polynomials (weight
     * x^alpha e^{-x}, alpha = k-1); the weights are the squared first components
     * of the corresponding normalized eigenvectors.
     */
    private void gaussLaguerreCategories(double R, double k,
                                         double[] lambdaOut, double[] weightOut) {
        int J = getNTypes();

        // Degenerate / near-deterministic case: point mass at R.
        if (Double.isInfinite(k) || k > 1e5) {
            for (int j = 0; j < J; j++) {
                lambdaOut[j] = R;
                weightOut[j] = 1.0 / J;
            }
            return;
        }

        double alpha = k - 1.0;   // weight function x^alpha e^{-x}
        double theta = R / k;     // Gamma scale

        // Symmetric tridiagonal Jacobi matrix:
        //   diagonal_i    = 2i + alpha + 1          (i = 0..J-1)
        //   offdiagonal_i = sqrt(i * (i + alpha))   (i = 1..J-1)
        double[][] jac = new double[J][J];
        for (int i = 0; i < J; i++) {
            jac[i][i] = 2.0 * i + alpha + 1.0;
            if (i > 0) {
                double off = Math.sqrt(i * (i + alpha));
                jac[i][i - 1] = off;
                jac[i - 1][i] = off;
            }
        }

        EigenDecomposition eig =
                new EigenDecomposition(new Array2DRowRealMatrix(jac, false));

        double wSum = 0.0;
        for (int j = 0; j < J; j++) {
            double node = eig.getRealEigenvalue(j);
            double v0 = eig.getEigenvector(j).getEntry(0);
            lambdaOut[j] = theta * Math.max(node, 0.0);  // guard tiny negative node
            weightOut[j] = v0 * v0;                       // Golub-Welsch weight
            wSum += weightOut[j];
        }
        // Renormalise so the weights sum to exactly 1 (guards rounding).
        for (int j = 0; j < J; j++)
            weightOut[j] /= wSum;

        // Sort categories by ascending rate (insertion sort over the small J
        // arrays, carrying the weights along) so the type index tracks
        // transmissibility: state 0 = lowest lambda, state J-1 = highest,
        // matching the equal-probability scheme. Category labels are
        // exchangeable, so this does not change the model.
        for (int a = 1; a < J; a++) {
            double lv = lambdaOut[a], wv = weightOut[a];
            int b = a - 1;
            while (b >= 0 && lambdaOut[b] > lv) {
                lambdaOut[b + 1] = lambdaOut[b];
                weightOut[b + 1] = weightOut[b];
                b--;
            }
            lambdaOut[b + 1] = lv;
            weightOut[b + 1] = wv;
        }
    }

    /**
     * Diagnostic: prints the Gamma discretisation (edges, per-type weights and
     * rates, moment summary, cross-birth row sums) at the given time, using the
     * model's actual computed arrays. Called once from initAndValidate so the
     * values are easy to check in the run log.
     */
    private void printDiscretizationDebug(double time) {
        int J = getNTypes();

        // populate the model's actual rate arrays for this time
        recalculateRates(time);

        double R   = meanRInput.get().getValuesAtTime(time)[0];
        double k   = kInput.get().getValuesAtTime(time)[0];
        double psi = samplingPropInput.get().getValuesAtTime(time)[0];
        double nu  = removalRateInput.get().getValuesAtTime(time)[0];

        boolean gl = gaussLaguerreInput.get();

        // Bin edges only exist for the equal-probability scheme.
        double[] edges = new double[J + 1];
        if (!gl && !(Double.isInfinite(k) || k > 1e5)) {
            double shape = k, scale = R / k;
            GammaDistribution gamma = new GammaDistribution(shape, scale, INV_CDF_ACCURACY);
            for (int i = 0; i <= J; i++) {
                double prob = (double) i / J;
                edges[i] = (prob == 0.0) ? 0.0
                        : (prob == 1.0) ? Double.POSITIVE_INFINITY
                        : gamma.inverseCumulativeProbability(prob);
            }
        }

        // Use the model's ACTUAL category weights (equal 1/J, or Gauss-Laguerre).
        double sumW = 0, wMean = 0, wM2 = 0;
        for (int j = 0; j < J; j++) {
            sumW  += categoryWeights[j];
            wMean += categoryWeights[j] * lambdaValues[j];
            wM2   += categoryWeights[j] * lambdaValues[j] * lambdaValues[j];
        }
        double wVar = wM2 - wMean * wMean;

        StringBuilder sb = new StringBuilder();
        sb.append("\n==== NegativeBinomialApproximation debug ====\n");
        sb.append("scheme = ").append(gl
                ? "Gauss-Laguerre quadrature (unequal weights)"
                : "equal-probability bins (equal weights)").append("\n");
        sb.append("time = ").append(time).append("\n");
        sb.append("R = ").append(R).append(", k = ").append(k)
          .append(", psi = ").append(psi).append(", nu = ").append(nu)
          .append(", J = ").append(J).append("\n");
        if (!gl) {
            sb.append("-- edges --\n");
            for (int i = 0; i <= J; i++)
                sb.append("edge[").append(i).append("] = ").append(edges[i]).append("\n");
        }
        sb.append("-- category rates and weights --\n");
        for (int j = 0; j < J; j++)
            sb.append("type ").append(j).append(": weight = ").append(categoryWeights[j])
              .append(", lambda = ").append(lambdaValues[j]).append("\n");
        sb.append("-- summary --\n");
        sb.append("sum(weights) = ").append(sumW).append("\n");
        sb.append("weighted mean(lambda) = ").append(wMean).append("\n");
        sb.append("weighted var(lambda) = ").append(wVar).append("\n");
        sb.append("target mean(lambda) = ").append(R).append("\n");
        sb.append("target var(lambda) = ").append(R * R / k).append("\n");
        sb.append("target offspring var = ").append(R + R * R / k).append("\n");
        sb.append("-- cross-birth row sums --\n");
        for (int j = 0; j < J; j++) {
            double rowSum = 0;
            for (int kk = 0; kk < J; kk++) rowSum += calculatedCrossBirthRates[j][kk];
            sb.append("type ").append(j).append(": rowSum = ").append(rowSum)
              .append(", expected nu*lambda = ").append(nu * lambdaValues[j]).append("\n");
        }
        sb.append("============================================");
        System.out.println(sb);
    }

    private void discretizeGamma(double R, double k, double[] lambdaOut) {
        int J = getNTypes();

        if (Double.isInfinite(k) || k > 1e5) {
            for (int j = 0; j < J; j++) {
                lambdaOut[j] = R;
            }
            return;
        }

        double shape = k;
        double scale = R / k;

        GammaDistribution gamma = new GammaDistribution(shape, scale, INV_CDF_ACCURACY);
        GammaDistribution gammaPlusOne = new GammaDistribution(shape + 1, scale, INV_CDF_ACCURACY);

        double[] edges = new double[J + 1];
        for (int i = 0; i <= J; i++) {
            double prob = (double) i / J;
            if (prob == 0.0) {
                edges[i] = 0.0;
            } else if (prob == 1.0) {
                edges[i] = Double.POSITIVE_INFINITY;
            } else {
                edges[i] = gamma.inverseCumulativeProbability(prob);
            }
        }

        for (int j = 0; j < J; j++) {
            double a = edges[j];
            double b = edges[j + 1];

            // CDFs at the bin edges. The top bin's upper edge is +infinity,
            // where both CDFs are exactly 1 (use the exact tail rather than
            // truncating at an arbitrary quantile, so the top bin also carries
            // exactly 1/J of the mass and captures the full superspreading tail).
            double cdfLoK  = gamma.cumulativeProbability(a);
            double cdfHiK  = Double.isInfinite(b) ? 1.0 : gamma.cumulativeProbability(b);
            double cdfLoK1 = gammaPlusOne.cumulativeProbability(a);
            double cdfHiK1 = Double.isInfinite(b) ? 1.0 : gammaPlusOne.cumulativeProbability(b);

            double denom = cdfHiK - cdfLoK;

            // Numerical guard: at small k the Gamma quantile function
            // underflows in the extreme lower tail, so adjacent edges can
            // collapse (b <= a) or enclose negligible probability mass
            // (denom ~ 0). num/denom is then 0/0 = NaN, which poisons the
            // birth-rate matrix. The limiting conditional mean of such a
            // degenerate low-probability bin is its (near-zero) lower edge.
            if (b <= a || denom < 1e-12) {
                lambdaOut[j] = a;
                continue;
            }

            double num = scale * shape * (cdfHiK1 - cdfLoK1);

            lambdaOut[j] = num / denom;
        }
    }


    // ===== VALIDATION =====

    @Override
    protected void validateParameterTypeCounts() {
        if (meanRInput.get().getNTypes() != 1) {
            throw new IllegalArgumentException("meanR should be scalar");
        }
        if (kInput.get().getNTypes() != 1) {
            throw new IllegalArgumentException("k should be scalar");
        }
        if (samplingPropInput.get().getNTypes() != 1) {
            throw new IllegalArgumentException("samplingProp should be scalar");
        }
        if (removalRateInput.get().getNTypes() != 1) {
            throw new IllegalArgumentException("removalRate should be scalar");
        }
    }


    // ===== LOGGING =====
    // Logs the per-type birth rate (nu * lambda_j) for each of the J
    // discretisation categories, evaluated at the start of the process.

    @Override
    public void init(PrintStream out) {
        String id = getID() == null ? "NBApprox" : getID();
        int J = getNTypes();
        for (int j = 0; j < J; j++)
            out.print(id + ".birthRate" + (j + 1) + "\t");
        for (int j = 0; j < J; j++)
            for (int k = 0; k < J; k++)
                out.print(id + ".crossBirth_" + (j + 1) + "_" + (k + 1) + "\t");
    }

    @Override
    public void log(long sample, PrintStream out) {
        recalculateRates(0.0);
        int J = getNTypes();
        for (int j = 0; j < J; j++)
            out.print(calculatedBirthRates[j] + "\t");
        for (int j = 0; j < J; j++)
            for (int k = 0; k < J; k++)
                out.print(calculatedCrossBirthRates[j][k] + "\t");
    }

    @Override
    public void close(PrintStream out) {
    }
}