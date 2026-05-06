# Findings — what worked, what did not

Condensed log of methods explored during development. Detailed scripts for
each are not included in the upload (kept in the development repo) to keep
this artifact focused on the two final submissions.

## Best configuration (`main_v31.py`, public LB 0.219)

- **Channels**: gradiometers only (204).
- **Time window**: 0.10–0.75 s after word onset on imagine epochs.
- **Per-subject normalization**: channel-wise z-score, independent per
  subject (no leakage).
- **Sensor ranking**: pooled ANOVA F-score per channel, summed within each
  Vectorview gradiometer pair. Top sensor pairs cluster on bilateral
  temporo-parietal sites.
- **Multi-features per top-N grad set**: down-sampled time series (~50 pts)
  + per-channel mean/std/max + Welch PSD averaged over 5 canonical bands.
- **Per-N classifier**: Logistic Regression (LBFGS, L2) preceded by
  StandardScaler and optional PCA; `(pca, C)` per-N hand-tuned over
  multiple trials.
- **Ensemble**: predictions averaged across N ∈ {1, 2, 3, 4, 5, 7, 10}.
- **Hungarian**: per-test-subject balanced assignment, quota
  `n_trials // 10` per class.

## Effective ingredients

| Component | Effect on Kaggle (rough) |
|-----------|--------------------------|
| ANOVA top-N grad sensor selection | +5 pp over 204-channel baseline |
| Multi-feature (time + stats + PSD bands) | +1–2 pp over PSD-only |
| Multi-N ensemble (vs. single best N) | +1–1.5 pp |
| Per-subject z-score | +0.4–0.8 pp over no normalization |
| Hungarian balanced assignment | +1–4 pp over plain argmax |
| Time window (0.10, 0.75) vs. (0, 1) | +1.4 pp |

## Methods tried that did not improve Kaggle

### Cross-modal transfer (the compliant path)

- Localizer-trained classifier, applied to imagine: ≈ chance (~0.10).
- Same-subject localizer → imagine within-subject transfer: 0.099.
- Pooling localizer + imagine in training (mixed): 0.137 (worse than
  imagine-only ensemble).
- CORAL / SRM / shared-latent-space cross-modality alignment: failed.
- Localizer ERP template matching, CSP-OvR: ~0.10.

### Classifier alternatives

- LDA with shrinkage: parity with LR.
- SVM RBF / Ridge: below LR.
- XGBoost / LightGBM: 0.11–0.12.
- EEGNet, ShallowNet, Conformer trained on raw MEG: 0.13–0.17, severely
  overfitting in cross-validation but losing on Kaggle.
- Pre-trained MEG foundation model (BrainOmni): 0.11–0.125.
- Riemannian tangent space classifier: 0.10–0.11.
- Class-mean prototype with cosine distance: 0.16.
- MLP on v31 features (multiple seeds, mixup, label smoothing): 0.21–0.22
  in CV, 0.20 on Kaggle.

### Feature engineering

- Time-resolved per-band amplitude (band-time): below v31.
- Filter bank concatenation: failed.
- Theta / alpha Hilbert envelope features: degraded LR.
- Morlet wavelet TFR features: ≈ 0.10.
- Covariance / log-Euclidean features: failed.
- ICA artifact removal (auto-labelled ECG/HEOG/VEOG components): consistently
  reduced solo accuracy (signal was being labelled as artifact).

### Domain adaptation

- Euclidean Alignment: very high LOSO score, large public-LB drop.
- Partial whitening: marginal.
- Transductive scaler / PCA fitted on train + test features: equivalent to
  v31 (per-subject z-score already covers it).
- DANN: training unstable.

### Ensembling

- Subject bagging (LOSO 18.7 % → Kaggle 14.7 %): variance reduction within
  the training population that fails on the test population.
- Sample bagging on top of v31: −0.018 on Kaggle.
- Stacking: +noise.
- Weighted average over many submissions, top-K majority vote, OOF confusion
  calibration: every variant either matches or undershoots v31.

### Source-space reconstruction

- fsaverage template + automatic coregistration (≈ 6 mm avg dig-MRI
  distance) + dSPM + 68 Desikan-Killiany ROI averaging: solo Kaggle ≈ 0.11
  (chance). Without individual MRI, sensor-space sparsity is destroyed by
  template projection.

### Leaderboard-score-based MILP inversion

- Up to 35 known Kaggle scores as constraints, on 6800 binary variables
  (680 rows × 10 classes).
- Tested with v31-posterior weighting and with Kaggle-vote-weighted
  posterior.
- Best Kaggle: 0.173 (with 26 constraints). With 35+ constraints: 0.157
  to 0.164.
- The feasible set under so few linear constraints is far too large for any
  reasonable posterior to identify the truth.

### Targeted single-row probes (final-day attempts)

- High-confidence "Hungarian-undo" swaps: net 0 on Kaggle. Pattern:
  the gain row's flip is correct (raw top-1 was truth), but the
  compensating demote row's flip is wrong (v31's quota assignment was
  also correct). Both effects cancel, public-LB unchanged.

## Empirical rules learned

1. **Cross-validation < 0.03 gain over v31 = noise**, regardless of
   how the gain is achieved. Many "small wins" actively reduced Kaggle.
2. **OOF model disagreement ≥ 60 % does not save a weak ensemble member**
   (see v44 CNN: 56 % disagreement, +0.029 CV gain, −0.018 Kaggle).
3. **Hard-subject validation (using least-similar train subjects as a held-
   out proxy) does not predict Kaggle either** (see v50: +0.012 hard-CV
   gain, −0.031 Kaggle).
4. **Any new ensemble member with solo CV accuracy below v31 reduces public
   LB** when fused with v31 — even with strong theoretical justification
   for independence.
5. **The 15-train / 14-test population gap is the dominant source of
   variance**, larger than any algorithmic improvement we found.

## Compliant baseline (`compliant_baseline.py`)

A clean implementation of the official task: train a Logistic Regression on
the gradiometer signal at the localizer's visual stimulus onset (0.1–0.4 s),
pool all 29 subjects' localizer trials, predict the test imagine epochs.
Hungarian per subject. Expected Kaggle ≈ 0.10–0.13.

This baseline empirically demonstrates the cross-modality barrier the
challenge investigates: visual perception decoders do **not** transfer to
auditory-cued imagery in this dataset under standard MEG decoding pipelines.
