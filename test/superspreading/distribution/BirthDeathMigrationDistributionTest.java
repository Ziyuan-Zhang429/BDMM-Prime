/*
 * Copyright (C) 2026
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

package superspreading.distribution;

import bdmmprime.parameterization.SkylineVectorParameter;
import bdmmprime.parameterization.TypeSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;
import org.junit.Test;
import superspreading.parameterization.NBTypeFrequencies;
import superspreading.parameterization.NegativeBinomialApproximation;

import static org.junit.Assert.assertEquals;

/**
 * Checks that the extension's tree prior agrees with BDMM-Prime's for all
 * combinations of its speed-ups, and that the edge cache is exact across
 * store/restore.
 */
public class BirthDeathMigrationDistributionTest {

    // Serially sampled tree; tip types are unknown (latent NB categories).
    private static final String NEWICK = "((((t0:0.4,t1:0.9):0.3,t2:0.5):0.6,(t3:0.2,t4:1.1):0.8):0.5,"
            + "((t5:0.7,t6:0.3):0.4,t7:1.2):0.9):0.0;";
    private static final String UNKNOWN_TYPES = "t0=?,t1=?,t2=?,t3=?,t4=?,t5=?,t6=?,t7=?";

    private static SkylineVectorParameter sky(String values, String changeTimes, RealParameter processLength) {
        SkylineVectorParameter s = new SkylineVectorParameter();
        if (changeTimes == null)
            s.initByName("skylineValues", new RealParameter(values), "processLength", processLength,
                    "typeSet", new TypeSet(1), "isScalar", true);
        else
            s.initByName("skylineValues", new RealParameter(values), "changeTimes", new RealParameter(changeTimes),
                    "processLength", processLength, "typeSet", new TypeSet(1), "isScalar", true);
        return s;
    }

    private static NegativeBinomialApproximation nbParam(boolean gaussLaguerre, String k,
                                                         String meanR, String meanRChangeTimes) {
        RealParameter origin = new RealParameter("3.0");
        NegativeBinomialApproximation p = new NegativeBinomialApproximation();
        p.initByName(
                "nTypes", 5,
                "gaussLaguerre", gaussLaguerre,
                "typeSet", new TypeSet(1),
                "processLength", origin,
                "meanR", sky(meanR, meanRChangeTimes, origin),
                "k", sky(k, null, origin),
                "samplingProp", sky("0.2", null, origin),
                "removalRate", sky("10.0", null, origin));
        return p;
    }

    private static Tree tree() {
        return new TreeParser(NEWICK, false, false, true, 0);
    }

    private static TraitSet unknownTypes(Tree tree) {
        TraitSet traitSet = new TraitSet();
        traitSet.initByName("traitname", "type", "taxa", tree.getTaxonset(), "value", UNKNOWN_TYPES);
        return traitSet;
    }

    private static double referenceLogP(Tree tree, NegativeBinomialApproximation p) {
        bdmmprime.distribution.BirthDeathMigrationDistribution density =
                new bdmmprime.distribution.BirthDeathMigrationDistribution();
        density.initByName(
                "parameterization", p,
                "startTypePriorProbs", nbFrequencies(p),
                "tree", tree,
                "typeTraitSet", unknownTypes(tree),
                "parallelize", false);
        return density.calculateLogP();
    }

    private static BirthDeathMigrationDistribution density(Tree tree, NegativeBinomialApproximation p,
                                                           boolean edgeCaching, boolean fastIntegrator,
                                                           boolean meanField) {
        BirthDeathMigrationDistribution density = new BirthDeathMigrationDistribution();
        density.initByName(
                "parameterization", p,
                "startTypePriorProbs", nbFrequencies(p),
                "tree", tree,
                "typeTraitSet", unknownTypes(tree),
                "parallelize", false,
                "useEdgeCaching", edgeCaching,
                "useFastIntegrator", fastIntegrator,
                "useMeanFieldODE", meanField);
        return density;
    }

    private static NBTypeFrequencies nbFrequencies(NegativeBinomialApproximation p) {
        NBTypeFrequencies frequencies = new NBTypeFrequencies();
        frequencies.initByName("parameterization", p);
        return frequencies;
    }

    @Test
    public void testMatchesBDMMPrime() {
        String[][] cases = {
                // gaussLaguerre, k, meanR, meanR change times
                {"false", "0.5", "1.6", null},
                {"true", "0.5", "1.6", null},
                {"true", "0.06", "1.6", null},
                {"true", "0.3", "1.6 0.8", "1.5"},
        };
        for (String[] c : cases) {
            NegativeBinomialApproximation p = nbParam(Boolean.parseBoolean(c[0]), c[1], c[2], c[3]);
            Tree tree = tree();
            double reference = referenceLogP(tree, p);

            for (boolean edgeCaching : new boolean[] {false, true})
                for (boolean fastIntegrator : new boolean[] {false, true})
                    for (boolean meanField : new boolean[] {false, true}) {
                        double logP = density(tree, p, edgeCaching, fastIntegrator, meanField).calculateLogP();
                        assertEquals(String.format("gaussLaguerre=%s k=%s meanR=%s cache=%s fast=%s meanField=%s",
                                        c[0], c[1], c[2], edgeCaching, fastIntegrator, meanField),
                                reference, logP, 1e-6 * (1.0 + Math.abs(reference)));
                    }
        }
    }

    @Test
    public void testEdgeCacheIsExactAcrossStoreRestore() {
        NegativeBinomialApproximation p = nbParam(true, "0.3", "1.6 0.8", "1.5");
        Tree tree = tree();
        BirthDeathMigrationDistribution cached = density(tree, p, true, true, false);

        double initial = cached.calculateLogP();

        // Tree move: change the height of an internal node, then reject it.
        Node node = tree.getNode(tree.getLeafNodeCount() + 1);
        double oldHeight = node.getHeight();
        double newHeight = 0.5 * (oldHeight + Math.max(node.getLeft().getHeight(), node.getRight().getHeight()));

        cached.store();
        node.setHeight(newHeight);
        double proposed = cached.calculateLogP();
        double proposedUncached = density(tree, p, false, true, false).calculateLogP();
        assertEquals("cached value after tree move", proposedUncached, proposed, 0.0);

        cached.restore();
        node.setHeight(oldHeight);
        assertEquals("value after restoring tree move", initial, cached.calculateLogP(), 0.0);

        // Parameter move: change k, then reject it.
        RealParameter k = (RealParameter) ((SkylineVectorParameter) p.kInput.get()).skylineValuesInput.get();
        double oldK = k.getValue();
        cached.store();
        k.setValue(0.1);
        p.kInput.get().initAndValidate();
        p.initAndValidate();
        double proposedK = cached.calculateLogP();
        double proposedKUncached = density(tree, p, false, true, false).calculateLogP();
        assertEquals("cached value after parameter move", proposedKUncached, proposedK, 0.0);

        cached.restore();
        k.setValue(oldK);
        p.kInput.get().initAndValidate();
        p.initAndValidate();
        assertEquals("value after restoring parameter move", initial, cached.calculateLogP(), 0.0);
    }
}
