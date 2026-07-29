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
import org.apache.commons.math3.distribution.GammaDistribution;

/**
 * Parameterization that approximates negative binomial offspring distribution
 * through discretized Gamma heterogeneity in transmission rates.
 */
public class NegativeBinomialApproximation extends Parameterization {

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
            20);

    // ===== DEBUG SETTINGS =====

    private static final double TAIL_PROB = 0.9999999999;
    private boolean debugPrinted = false;

    // ===== CALCULATED VALUES =====

    private double[] lambdaValues;
    private double[] lambdaWeights;
    private double[] lambdaEdges;

    private double[] calculatedBirthRates;
    private double[][] calculatedCrossBirthRates;
    private double[] calculatedDeathRates;
    private double[] calculatedSamplingRates;

    // ===== INITIALIZATION =====

    //    @Override
//    public void initAndValidate() {
//        super.initAndValidate();

//        int requestedTypes = nTypesInput.get();
//        if (requestedTypes < 2) {
//            throw new IllegalArgumentException(
//                    "NegativeBinomialApproximation requires nTypes >= 2");
//        }

//        nTypes = requestedTypes;
//    }

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
        recalculateRates(time);
        return calculatedCrossBirthRates;
    }

    @Override
    protected double[] getDeathRateValues(double time) {
        recalculateRates(time);
        return calculatedDeathRates;
    }

    @Override
    protected double[] getSamplingRateValues(double time) {
        recalculateRates(time);
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

    private void recalculateRates(double time) {
        int J = getNTypes();

        if (lambdaValues == null) {
            lambdaValues = new double[J];
            lambdaWeights = new double[J];
            lambdaEdges = new double[J + 1];

            calculatedBirthRates = new double[J];
            calculatedCrossBirthRates = new double[J][J];
            calculatedDeathRates = new double[J];
            calculatedSamplingRates = new double[J];
        }

        double meanR = meanRInput.get().getValuesAtTime(time)[0];
        double k = kInput.get().getValuesAtTime(time)[0];
        double psi = samplingPropInput.get().getValuesAtTime(time)[0];
        double nu = removalRateInput.get().getValuesAtTime(time)[0];

        discretizeGamma(meanR, k, lambdaValues, lambdaWeights, lambdaEdges);

        for (int j = 0; j < J; j++) {
            calculatedBirthRates[j] = nu * lambdaValues[j] / (double) J;

            for (int kk = 0; kk < J; kk++) {
                calculatedCrossBirthRates[j][kk] = nu * lambdaValues[j] / (double) J;
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

        if (!debugPrinted) {
            printDiscretizationDebug(time, meanR, k, psi, nu);
            debugPrinted = true;
        }
    }

    private void discretizeGamma(double R, double k,
                                 double[] lambdaOut,
                                 double[] weightOut,
                                 double[] edgeOut) {
        int J = getNTypes();

        if (Double.isInfinite(k) || k > 1e5) {
            for (int j = 0; j < J; j++) {
                lambdaOut[j] = R;
                weightOut[j] = 1.0 / (double) J;
            }
            for (int i = 0; i < J; i++) {
                edgeOut[i] = Double.NaN;
            }
            edgeOut[J] = Double.POSITIVE_INFINITY;
            return;
        }

        double shape = k;
        double scale = R / k;

        GammaDistribution gamma = new GammaDistribution(shape, scale);
        GammaDistribution gammaPlusOne = new GammaDistribution(shape + 1.0, scale);

        for (int i = 0; i <= J; i++) {
            double prob = (double) i / (double) J;
            if (prob == 0.0) {
                edgeOut[i] = 0.0;
            } else if (prob == 1.0) {
                edgeOut[i] = Double.POSITIVE_INFINITY;
            } else {
                edgeOut[i] = gamma.inverseCumulativeProbability(prob);
            }
        }

        for (int j = 0; j < J; j++) {
            double a = edgeOut[j];
            double b = edgeOut[j + 1];

            double bUsed = b;
            if (Double.isInfinite(bUsed)) {
                bUsed = gamma.inverseCumulativeProbability(TAIL_PROB);
            }

            double denom = gamma.cumulativeProbability(bUsed)
                    - gamma.cumulativeProbability(a);

            double num = scale * shape
                    * (gammaPlusOne.cumulativeProbability(bUsed)
                    - gammaPlusOne.cumulativeProbability(a));

            lambdaOut[j] = num / denom;
            weightOut[j] = denom;
        }
    }

    private void printDiscretizationDebug(double time, double meanR, double k, double psi, double nu) {
        int J = getNTypes();

        System.out.println("==== NegativeBinomialApproximation debug ====");
        System.out.println("time = " + time);
        System.out.println("R = " + meanR
                + ", k = " + k
                + ", psi = " + psi
                + ", nu = " + nu
                + ", J = " + J
                + ", tailProb = " + TAIL_PROB);

        System.out.println("-- edges --");
        for (int i = 0; i < lambdaEdges.length; i++) {
            System.out.println("edge[" + i + "] = " + lambdaEdges[i]);
        }

        System.out.println("-- category rates and weights --");
        double sumW = 0.0;
        for (int j = 0; j < J; j++) {
            System.out.println("type " + j
                    + ": weight = " + lambdaWeights[j]
                    + ", lambda = " + lambdaValues[j]);
            sumW += lambdaWeights[j];
        }

        double mean = 0.0;
        for (int j = 0; j < J; j++) {
            mean += lambdaWeights[j] * lambdaValues[j];
        }
        mean /= sumW;

        double var = 0.0;
        for (int j = 0; j < J; j++) {
            double diff = lambdaValues[j] - mean;
            var += lambdaWeights[j] * diff * diff;
        }
        var /= sumW;

        System.out.println("-- summary --");
        System.out.println("sum(weights) = " + sumW);
        System.out.println("weighted mean(lambda) = " + mean);
        System.out.println("weighted var(lambda) = " + var);
        System.out.println("target mean(lambda) = " + meanR);

        if (!Double.isInfinite(k) && k <= 1e5) {
            System.out.println("target var(lambda) = " + (meanR * meanR / k));
            System.out.println("target offspring var = " + (meanR + meanR * meanR / k));
        }

        System.out.println("-- cross-birth row sums --");
        for (int j = 0; j < J; j++) {
            double rowSum = 0.0;
            for (int kk = 0; kk < J; kk++) {
                rowSum += calculatedCrossBirthRates[j][kk];
            }

            System.out.println("type " + j
                    + ": rowSum = " + rowSum
                    + ", expected nu*lambda = " + (nu * lambdaValues[j]));
        }

        System.out.println("============================================");
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
}