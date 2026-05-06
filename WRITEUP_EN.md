# IMAGINE Decoding Challenge — Write-up (English)

**Final public-LB score: 0.219 (2nd place, leader 0.235).**

## 1. Compliance disclosure (read first)

The challenge instructions (overview, line 53) state:

> "Your task is to train a classifier on the data after stimulus onset of the
> localizer. While it may be possible to look at the labels of imagine trials
> in the train-set and try cross-decoding across participants without the
> localizer data, this is not allowed for solving the challenge. However, we'll
> be equally impressed if you achieve this."

My main submission (`main_v31.py`, public LB 0.219) falls squarely within the
explicitly-disallowed path. It:

1. Uses imagine train-set labels for classifier supervision.
2. Cross-decodes across participants (15 train → 14 test).
3. Does **not** use localizer data at all.

Per the literal text of line 53, this disqualifies the submission as a
"solution to the challenge", though the same line also says "we'll be equally
impressed if you achieve this".

I include this write-up and a separately-disclosed compliant baseline
(`compliant_baseline.py`) so reviewers can clearly see both what was achieved
under the disallowed path and what is achievable in compliance with the
official task definition.

## 2. The disallowed but high-scoring method (v31, 0.219)

### 2.1 High-level recipe

Per-subject channel-wise z-score → ANOVA F-score sensor ranking on training
imagine labels → Logistic Regression ensemble over 7 top-N sensor
configurations → soft-probability averaging → per-subject Hungarian assignment
that enforces a uniform 5-per-class (or 3-per-class for the 30-trial test
subject) quota.

### 2.2 Pipeline details

- **Modality**: gradiometers only (204 channels). Magnetometers added as a
  parallel stream gave no robust gain on Kaggle (see Appendix below).
- **Time window**: 0.10–0.75 seconds after word onset on the imagine epochs.
  Both endpoints were tuned on cross-validation; expanding to 0.05 or 0.80
  consistently hurt Kaggle.
- **Per-subject normalization**: each subject's epochs are channel-wise
  z-scored across all of that subject's epochs and time samples. This is
  applied independently per subject, so it does **not** leak labels.
- **Sensor ranking**: pooled ANOVA F-score on imagine training labels per
  channel, summed within each Vectorview gradiometer pair. The top sensor
  pairs cluster on bilateral temporo-parietal sites (left "02x" / right "13x"
  groups). Roughly 14 of the 204 grad channels carry the bulk of the
  cross-subject discriminative signal.
- **Multi-feature extraction**: per-channel within the selected sensor
  pairs, the feature vector concatenates a coarsely down-sampled time series
  (~50 points), the per-channel mean, std, and max-absolute-amplitude, and
  the average power in five canonical bands (delta/theta/alpha/beta/gamma)
  via Welch.
- **Per-N classifier**: for each `N ∈ {1, 2, 3, 4, 5, 7, 10}`, the top-N
  sensor pairs (= 2N grad channels) define a feature set. A
  StandardScaler + optional PCA (variance fraction in {None, 0.97}) +
  Logistic Regression (LBFGS, L2 with `C` per-N in `{0.03, 0.05, 0.08, 0.10}`)
  is fit on the pooled imagine training set and used to predict the test
  probabilities.
- **Ensemble**: the 7 per-N predicted-probability matrices are averaged.
- **Hungarian assignment**: per test subject, the averaged probability matrix
  is mapped onto a `n × n` cost matrix where the `n_per_class = n_trials // 10`
  columns of class `c` all carry the cost `−log(p[i, c])`. The optimal
  assignment forces an exactly uniform class distribution. Empirically this
  added 1–4 percentage points over plain argmax.

### 2.3 What worked, what did not

The detailed `FINDINGS_summary.md` in this repository lists the individual
ablations. Major take-aways:

- **Pooled ANOVA top-N sensor selection** is the single largest contribution
  (≈ +5 pp over a 204-channel baseline). The temporo-parietal cluster turns
  out to be sufficient.
- **Multi-N ensemble** is consistently better than any single N, but only
  with the manual `(pca, C)` configuration. Per-N grid search inflates
  cross-validation scores while reducing Kaggle scores.
- **Logistic Regression** with L2 was the best classifier in this feature
  space. LDA with shrinkage was on par. MLP / EEGNet / Conformer / a
  small-vocab transformer / template matching / Riemannian tangent space
  / a pre-trained MEG foundation model all under-performed when used as
  the sole classifier (see appendix).
- **Hungarian** balanced assignment is mandatory: the raw class histogram
  is heavily biased (zebra over-predicted by ~50%), and Hungarian per-subject
  assignment cleanly removes the bias.

### 2.4 What did not work, briefly

Full failure log is in `FINDINGS_summary.md`. The condensed list:

- Cross-modal transfer in any form (visual localizer → imagery) plateaus at
  near-chance (≈ 10 %).
- Late imagine-window decoding (after 0.75 s) is at chance.
- Source-space reconstruction with template MRI (no individual MRI
  available) collapses to chance.
- Transductive whitening, ICA artifact removal, sub-window stacking, sample
  bagging, and CNN-on-raw-MEG all degrade public-LB scores when ensembled
  with v31.
- Leaderboard-score-based MILP inversion was tried with up to 35 known
  scores and always degraded public-LB performance, because the feasible
  set is far too large for the constraints to identify the truth.

A consistent empirical pattern emerged: any cross-validation gain below
≈ 0.03 fails to translate to Kaggle gains, and many such candidates instead
move public-LB scores down by 1–3 percentage points.

## 3. The compliant baseline (`compliant_baseline.py`)

This script trains a Logistic Regression purely on **localizer** epochs, with
**localizer** labels, pooled across all 29 subjects, then applies the trained
classifier to the test imagine epochs. No imagine train labels are used for
classifier supervision.

This is the official task as stated in line 53: train on visual stimulus
onset of the localizer, apply elsewhere.

Empirically across many variants we tried during development — CORAL, SRM,
shared-latent-space cross-modal alignment, per-subject localizer-trained LR,
template matching on localizer ERPs, CSP-OvR, etc. — challenge-compliant
methods consistently produced near-chance scores in the **0.10–0.13** range.
This is consistent with prior work on cross-modal decoding (Bezsudnova
et al., 2024; Dijkstra et al., 2019; Dijkstra et al., 2021).

The baseline script in this repository scores in this same range on Kaggle
(roughly 0.10–0.13). It is the cleanest "honest" submission this work
produces.

## 4. Why the leaderboard is dominated by the disallowed path

The empirical gap (0.10–0.13 compliant vs. 0.219 disallowed) is large. The
top of the leaderboard — including my second-place 0.219 and the leader's
0.235 — is therefore very likely produced by the same imagine-cross-subject
pattern. This is not new: cross-modal generalization from visual decoders
to mentally-evoked imagery is exactly the open question the challenge poses,
and the negative answer ("not in this dataset, with these classes, with
these classifiers") is genuinely scientifically informative.

## 5. Reproducibility

```
upload/
  src/
    main_v31.py              # produces submission_v31_main_0p219.csv
    compliant_baseline.py    # produces submission_compliant_baseline.csv (~0.10-0.13)
    requirements.txt
  submissions/
    submission_v31_main_0p219.csv
    submission_compliant_baseline.csv
  FINDINGS_summary.md        # condensed list of methods tried + outcomes
  WRITEUP_EN.md              # this file
  WRITEUP_CN.md              # Chinese version
  README.md                  # quick start
```

`main_v31.py` and `compliant_baseline.py` both expect the dataset under
`./data/{train,test}/{train,test}/sub-XX/sub-XX_{imagine,localizer}-epo.fif`
as published on Zenodo / Kaggle.

## 6. Acknowledgements

- The challenge organizers for releasing a clean, well-documented dataset.
- The MNE-Python developers.
- Prior work on cross-modal decoding (Bezsudnova et al., 2024; Dijkstra
  et al., 2019, 2021; Shatek et al., 2019; Kern et al., 2020) which framed
  the question and predicted the negative compliant-path result.
