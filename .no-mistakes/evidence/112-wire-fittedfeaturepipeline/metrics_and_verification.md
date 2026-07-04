# Issue #112 - Wire FittedFeaturePipeline into frequency training

Behavior-preserving refactor: `train_and_evaluate()` in `src/train/frequency.py`
now fits a single `FittedFeaturePipeline` on the train split and transforms both
splits through it, replacing the manual `Vehicle`/`Driver` instantiation.

## Behavior preservation (old path vs new path)

Ran both the pre-#112 manual sequence (`Vehicle().fit().transform()` then
`Driver().transform()`) and the new `FittedFeaturePipeline` on the same
train/test split, then asserted frame equality with `pandas.testing.assert_frame_equal`:

```
OK: train frame identical  shape (84444, 36)
OK: test  frame identical  shape (21111, 36)
OK: transform-before-fit guarded -> FittedFeaturePipeline must be fitted before transform; call fit() first.
```

Both frames are byte-for-byte identical, confirming the refactor changes no
behavior. The fit-before-transform guard raises `RuntimeError` as documented.

## Real training run through the new pipeline (`scripts.gate_metrics_smoke`)

The frequency model (`poisson_regressor`) trains end-to-end through the new
pipeline on the committed dataset via `config/smoke.yaml`:

```
Frequency model metrics (poisson_regressor)

Metric                       Train            Test
--------------------------------------------------
MSE                       1.216812        1.238017
RMSE                      1.103092        1.112662
MAE                       0.641593        0.645238
R2                        0.000000       -0.000001
MedAE                     0.394463        0.394463
Poisson deviance          1.510919        1.529094
```

Diagnostic plots produced by the run are saved alongside this file:
`claims_distribution.png`, `feature_importance.png`, `residuals.png`,
`actual_vs_predicted.png`.
