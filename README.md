# Superspreading

Superspreading is a [BEAST 2](http://www.beast2.org/) package that extends
[BDMM-Prime](https://github.com/tgvaughan/BDMM-Prime) with a negative-binomial
model of superspreading, in which individual reproductive numbers follow a
Gamma distribution with mean `meanR` and dispersion `k`.

The package does not replace BDMM-Prime; it depends on it and adds:

- `superspreading.parameterization.NegativeBinomialApproximation`: a BDMM-Prime
  parameterization that discretises the Gamma-distributed reproductive number
  into `nTypes` latent categories (equal-probability bins, or Gauss-Laguerre
  quadrature with `gaussLaguerre="true"`).
- `superspreading.parameterization.NBTypeFrequencies`: the category weights of
  a `NegativeBinomialApproximation`, for use as `startTypePriorProbs`.
- `superspreading.distribution.BirthDeathMigrationDistribution`: a drop-in
  replacement for BDMM-Prime's tree prior with the same inputs and faster
  likelihood evaluation:
  - `useEdgeCaching` (default `true`): only edges whose subtree, edge times or
    initial conditions changed are re-integrated after a proposal.
  - `useFastIntegrator` (default `true`): a lean Dormand-Prince 5(4) integrator
    (same method and tolerances as BDMM-Prime's) and faster ODE derivatives.
  - `useMeanFieldODE` (default `false`): O(J) instead of O(J^2) ODEs for rank-1
    parameterizations without migration, such as `NegativeBinomialApproximation`.
  - `checkEdgeCaching` (default `false`): debugging aid that recomputes every
    evaluation without the cache and reports any mismatch.

All speed-ups are exact up to the ODE tolerance.

## Usage

Use the classes of this package in place of the BDMM-Prime ones, e.g.

```xml
<parameterization id="param" spec="superspreading.parameterization.NegativeBinomialApproximation"
                  nTypes="5" gaussLaguerre="true" typeSet="@typeSet"> ... </parameterization>

<distribution id="bdmm" spec="superspreading.distribution.BirthDeathMigrationDistribution"
              tree="@tree" parameterization="@param" typeTraitSet="@typeTraitSet">
    <startTypePriorProbs spec="superspreading.parameterization.NBTypeFrequencies" parameterization="@param"/>
</distribution>
```

## Building from source

Requires OpenJDK 17 and Apache Ant. Running

```sh
ant
```

downloads BDMM-Prime and its dependencies into `deps/` and creates the package
in `dist/`. `ant install` also installs it into the local BEAST 2.7 package
directory, and `ant test` runs the unit tests.

## License

Superspreading is free software, distributed under the terms of version 3 of
the GNU General Public License (see `COPYING`).
