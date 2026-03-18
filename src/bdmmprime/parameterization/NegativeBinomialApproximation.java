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
            5);  // default value


    // ===== CALCULATED VALUES =====

    private double[] lambdaValues;
    private double[] calculatedBirthRates;
    private double[][] calculatedCrossBirthRates;
    private double[] calculatedDeathRates;
    private double[] calculatedSamplingRates;


    // ===== INITIALIZATION =====

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        // Set nTypes from input or use default
        int requestedTypes = nTypesInput.get();
        if (requestedTypes < 2) {
            throw new IllegalArgumentException(
                    "NegativeBinomialApproximation requires nTypes >= 2");
        }

        // This sets the nTypes in the parent class
        nTypes = requestedTypes;
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

    private void recalculateRates(double time) {
        int J = getNTypes();

        if (lambdaValues == null) {
            lambdaValues = new double[J];
            calculatedBirthRates = new double[J];
            calculatedCrossBirthRates = new double[J][J];
            calculatedDeathRates = new double[J];
            calculatedSamplingRates = new double[J];
        }

        double meanR = meanRInput.get().getValuesAtTime(time)[0];
        double k = kInput.get().getValuesAtTime(time)[0];
        double psi = samplingPropInput.get().getValuesAtTime(time)[0];
        double nu = removalRateInput.get().getValuesAtTime(time)[0];

        discretizeGamma(meanR, k, lambdaValues);

        for (int j = 0; j < J; j++) {
            calculatedBirthRates[j] = 0.0;

            for (int kk = 0; kk < J; kk++) {
                calculatedCrossBirthRates[j][kk] = nu * lambdaValues[j] / J;
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

        GammaDistribution gamma = new GammaDistribution(shape, scale);
        GammaDistribution gammaPlusOne = new GammaDistribution(shape + 1, scale);

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

            if (Double.isInfinite(b)) {
                b = gamma.inverseCumulativeProbability(0.9999);
            }

            double denom = gamma.cumulativeProbability(b) -
                    gamma.cumulativeProbability(a);

            double num = scale * shape *
                    (gammaPlusOne.cumulativeProbability(b) -
                            gammaPlusOne.cumulativeProbability(a));

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
}