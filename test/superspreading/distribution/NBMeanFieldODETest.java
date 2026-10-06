/*
 * Copyright (C) 2025
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import bdmmprime.distribution.P0GeSystem;
import bdmmprime.distribution.P0System;
import superspreading.parameterization.NegativeBinomialApproximation;
import bdmmprime.parameterization.SkylineVectorParameter;
import bdmmprime.parameterization.TypeSet;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Verifies that the mean-field ODE systems (NBP0System / NBP0GeSystem) are
 * numerically identical to the default O(J^2) systems (P0System / P0GeSystem)
 * for a NegativeBinomialApproximation parameterization, under both the
 * equal-weight and Gauss-Laguerre discretisation schemes.
 */
public class NBMeanFieldODETest {

    private static SkylineVectorParameter scalarSky(String value, RealParameter processLength) {
        SkylineVectorParameter s = new SkylineVectorParameter();
        s.initByName(
                "skylineValues", new RealParameter(value),
                "processLength", processLength,
                "typeSet", new TypeSet(1),
                "isScalar", true);
        return s;
    }

    private static NegativeBinomialApproximation nbParam(
            int nTypes, boolean gaussLaguerre, double origin, String k) {
        RealParameter originParam = new RealParameter(Double.toString(origin));
        NegativeBinomialApproximation p = new NegativeBinomialApproximation();
        p.initByName(
                "nTypes", nTypes,
                "gaussLaguerre", gaussLaguerre,
                "typeSet", new TypeSet(1),
                "processLength", originParam,
                "meanR", scalarSky("1.6", originParam),
                "k", scalarSky(k, originParam),
                "samplingProp", scalarSky("0.2", originParam),
                "removalRate", scalarSky("10.0", originParam));
        return p;
    }

    private static void assertClose(String msg, double expected, double actual) {
        assertEquals(msg, expected, actual, 1e-9 * (1.0 + Math.abs(expected)));
    }

    @Test
    public void testP0DerivativesMatch() {
        for (boolean gl : new boolean[]{false, true}) {
            NegativeBinomialApproximation p = nbParam(5, gl, 5.0, "0.5");
            P0System ref = new P0System(p, 1e-100, 1e-7);
            NBP0System mf = new NBP0System(p, 1e-100, 1e-7);

            assertEquals(ref.getDimension(), mf.getDimension());
            int dim = ref.getDimension();
            Random rng = new Random(20250728L);

            for (int interval = 0; interval < ref.nIntervals; interval++) {
                ref.setInterval(interval);
                mf.setInterval(interval);
                for (int trial = 0; trial < 25; trial++) {
                    double[] y = new double[dim];
                    for (int i = 0; i < dim; i++) y[i] = rng.nextDouble();
                    double[] dRef = new double[dim], dMf = new double[dim];
                    ref.computeDerivatives(0.3, y, dRef);
                    mf.computeDerivatives(0.3, y, dMf);
                    for (int i = 0; i < dim; i++)
                        assertClose("p0 gl=" + gl + " iv=" + interval + " i=" + i,
                                dRef[i], dMf[i]);
                }
            }
        }
    }

    @Test
    public void testP0GeDerivativesMatch() {
        for (boolean gl : new boolean[]{false, true}) {
            NegativeBinomialApproximation p = nbParam(5, gl, 5.0, "0.5");
            P0GeSystem ref = new P0GeSystem(p, 1e-100, 1e-7);
            NBP0GeSystem mf = new NBP0GeSystem(p, 1e-100, 1e-7);

            assertEquals(ref.getDimension(), mf.getDimension());
            int dim = ref.getDimension();
            Random rng = new Random(987654321L);

            for (int interval = 0; interval < ref.nIntervals; interval++) {
                ref.setInterval(interval);
                mf.setInterval(interval);
                for (int trial = 0; trial < 25; trial++) {
                    double[] y = new double[dim];
                    for (int i = 0; i < dim; i++) y[i] = rng.nextDouble();
                    double[] dRef = new double[dim], dMf = new double[dim];
                    ref.computeDerivatives(0.3, y, dRef);
                    mf.computeDerivatives(0.3, y, dMf);
                    for (int i = 0; i < dim; i++)
                        assertClose("p0ge gl=" + gl + " iv=" + interval + " i=" + i,
                                dRef[i], dMf[i]);
                }
            }
        }
    }

    @Test
    public void testLikelihoodMatches() {
        String newick = "((t0[&type=0]:1.0,t1[&type=0]:1.0):1.0,t2[&type=0]:2.0):0.0;";

        for (boolean gl : new boolean[]{false, true}) {
            double logPRef = bdmmLogP(newick, gl, false);
            double logPMeanField = bdmmLogP(newick, gl, true);
            assertEquals("likelihood gl=" + gl,
                    logPRef, logPMeanField, 1e-6 * (1.0 + Math.abs(logPRef)));
        }
    }

    private double bdmmLogP(String newick, boolean gaussLaguerre, boolean useMeanField) {
        NegativeBinomialApproximation p = nbParam(5, gaussLaguerre, 3.0, "0.5");
        BirthDeathMigrationDistribution density = new BirthDeathMigrationDistribution();
        density.initByName(
                "parameterization", p,
                "startTypePriorProbs", new RealParameter("0.2 0.2 0.2 0.2 0.2"),
                "conditionOnSurvival", false,
                "tree", new TreeParser(newick, false, false, true, 0),
                "typeLabel", "type",
                "parallelize", false,
                "useMeanFieldODE", useMeanField);
        return density.calculateLogP();
    }
}
