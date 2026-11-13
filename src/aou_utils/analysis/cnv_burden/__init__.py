"""CNV burden (Mb deleted / duplicated) vs quantitative traits such as height and BMI.

Modeled on the PGC CNV burden analyses (PGCCNV_GWAS_XD_2026, scripts_burden / scripts_r2), but with
linear models for quantitative traits: beta per Mb and incremental R² over covariates.

- `traits`: pull height / BMI medians and transform them (Box-Cox by sex, residualize, z-score).
- `models`: burden regressions (`burden_test`).
Per-person burden comes from `aou_utils.cnv.segments.burden`.

Import the modules directly: `from aou_utils.analysis.cnv_burden import models, traits`.
"""
