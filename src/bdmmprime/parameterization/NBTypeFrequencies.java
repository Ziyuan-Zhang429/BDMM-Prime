/*
 * Copyright (C) 2025
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package bdmmprime.parameterization;

import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;

/**
 * Exposes the category weights {w_j} of a {@link NegativeBinomialApproximation}
 * as a Function of dimension nTypes that sums to 1.
 *
 * Intended use: the BDMM `startTypePriorProbs` (the prior for the type of the
 * lineage at the origin). The category weights are the stationary type
 * distribution of the process, so using them as the origin prior makes the
 * founding lineage consistent with the rest of the epidemic. This matters for
 * the Gauss-Laguerre scheme, where the weights are strongly unequal: leaving
 * the origin prior at the default uniform 1/J lets the model place the origin
 * in a rare high-lambda category, which biases meanR downward. For the
 * equal-weight scheme the weights are 1/J, so this reduces to the uniform
 * default (harmless).
 *
 * Values are recomputed live from the parameterization, so they track the
 * current meanR/k during MCMC.
 */
public class NBTypeFrequencies extends CalculationNode implements Function {

    public Input<NegativeBinomialApproximation> parameterizationInput = new Input<>(
            "parameterization",
            "NegativeBinomialApproximation whose category weights to expose "
            + "as origin type frequencies.",
            Input.Validate.REQUIRED);

    private NegativeBinomialApproximation nb;

    @Override
    public void initAndValidate() {
        nb = parameterizationInput.get();
    }

    @Override
    public int getDimension() {
        return nb.getNTypes();
    }

    @Override
    public double getArrayValue() {
        return nb.getCategoryWeights()[0];
    }

    @Override
    public double getArrayValue(int i) {
        return nb.getCategoryWeights()[i];
    }
}
