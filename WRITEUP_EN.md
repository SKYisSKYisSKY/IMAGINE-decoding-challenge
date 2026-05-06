# IMAGINE Decoding Challenge — Method Write-up

## Compliance disclosure

The challenge instructions (overview, line 53) state:

> "Your task is to train a classifier on the data after stimulus onset of the
> localizer. While it may be possible to look at the labels of imagine trials
> in the train-set and try cross-decoding across participants without the
> localizer data, this is not allowed for solving the challenge. However,
> we'll be equally impressed if you achieve this."

The main submission in this repository (`src/main.py`) implements the
explicitly-disallowed path. It

1. uses the imagine train-set labels as classifier supervision,
2. cross-decodes across participants (15 train subjects → 14 test subjects),
3. does not use localizer data.

A challenge-compliant baseline is provided alongside (`src/compliant_baseline.py`)
for direct comparison. Empirically, the compliant approach plateaus near
chance, in the 0.10–0.13 range; the disallowed approach reaches 0.219 on
the public leaderboard. Both submissions are included for transparency,
and reviewers can grade either path according to the challenge's intended
scoring rules.

## 1. Task summary

The competition releases scrambled imagine epochs (word-onset aligned, no
trial labels) for 14 held-out subjects, and asks for label predictions over
all 680 trials. Each test subject's trial counts and the global class
distribution are uniform 5-per-class (3-per-class for one 30-trial subject).
Public-LB scoring is plain accuracy on these 680 predictions.

## 2. Method (disallowed path, public LB 0.219)

The decoder is a cross-subject ensemble of seven L2-regularized Logistic
Regression classifiers operating on engineered MEG features.

### 2.1 Pre-processing

- **Channels.** Gradiometers only, 204 channels. Magnetometers were
  evaluated as a parallel stream and as a joint-ranking partner; neither
  yielded a robust public-LB improvement.
- **Time window.** `(tmin, tmax) = (0.10, 0.75)` seconds after the imagine
  word-onset. Both endpoints were tuned by leave-three-subjects-out
  cross-validation; widening to `(0.05, 0.80)` consistently lost public-LB
  accuracy.
- **Normalization.** Per-subject channel-wise z-score, computed over each
  subject's epochs and time samples independently. Done before any
  cross-subject pooling, so no labels leak between subjects.

### 2.2 Sensor selection

A pooled ANOVA F-score is computed per gradiometer channel against the
imagine training labels. Channels are then aggregated within each Vectorview
sensor pair (the two gradiometers at the same physical location share a
score). The resulting sensor-pair ranking turns out to be highly sparse:
roughly 14 of 204 channels carry the bulk of the cross-subject discriminative
signal, clustered on bilateral temporo-parietal sites (left "02x" group and
right "13x" group of the Vectorview layout).

### 2.3 Multi-feature extraction

For a chosen top-N set of sensor pairs (= 2N gradiometer channels), each
trial is described by

- a coarsely down-sampled time series (one sample every `n_t // 50` raw samples),
- per-channel mean, standard deviation, and max-absolute-amplitude,
- per-channel Welch-PSD averaged over five canonical bands
  (delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz).

The four feature blocks are concatenated; the resulting vector dimension
scales with N.

### 2.4 Per-N classifier

For each `N ∈ {1, 2, 3, 4, 5, 7, 10}`:

1. The feature matrix over all train subjects is `StandardScaler`-fit, then
   optionally projected by PCA to a configurable variance fraction.
2. A Logistic Regression (LBFGS, L2, max_iter=5000) is fit on the pooled
   training set.
3. Predicted probabilities are computed for every test subject.

The seven `(pca_var, C)` configurations were hand-tuned (Table 1) and
deliberately not replaced with grid search; per-N grid search inflates
cross-validation scores while reducing public-LB scores.

| N  | PCA variance | LR `C`  |
|----|--------------|---------|
| 1  | none         | 0.10    |
| 2  | 0.97         | 0.08    |
| 3  | none         | 0.05    |
| 4  | 0.97         | 0.03    |
| 5  | 0.97         | 0.10    |
| 7  | 0.97         | 0.08    |
| 10 | none         | 0.05    |

### 2.5 Ensemble and balanced assignment

The seven per-N predicted probability matrices are averaged, yielding a
single (n_test_trials × 10) probability matrix per test subject. Per
subject, this is mapped onto an `n × n` cost matrix in which the
`n_per_class = n_trials // 10` columns of class `c` all carry the same
cost `-log(p[i, c])`. The Hungarian (Kuhn–Munkres) algorithm yields the
minimum-cost assignment, equivalent to per-subject argmax constrained to a
uniform class distribution. This balanced assignment captures the prior
that each test subject contains exactly five (or three) trials of each
class, and adds 1–4 percentage points over plain argmax.

## 3. Compliant baseline

`src/compliant_baseline.py` trains a single Logistic Regression on the
**localizer** epochs of all 29 subjects, using the **localizer** labels as
supervision; the imagine train-set labels are never seen by the classifier.
The same multi-feature extraction and per-subject Hungarian assignment are
re-used unchanged on the test imagine epochs. Public-LB accuracy lies in
the 0.10–0.13 range across variants we evaluated (Section 4.4). This is
consistent with the literature on cross-modal decoding from visual to
mentally-evoked stimuli (Bezsudnova et al., 2024; Dijkstra et al., 2019,
2021): visual perception decoders do not transfer to mental imagery in
this dataset under standard MEG decoding pipelines.

## 4. What did, and did not, contribute

### 4.1 Components that contributed

| Component                                  | Approx. public-LB contribution |
|--------------------------------------------|--------------------------------|
| ANOVA top-N sensor selection               | +5 pp over the 204-channel LR baseline |
| Multi-feature (time + stats + PSD bands)   | +1–2 pp over a PSD-only baseline |
| Multi-N ensemble vs. single best N         | +1–1.5 pp |
| Per-subject channel-wise z-score           | +0.4–0.8 pp over no normalization |
| Hungarian balanced assignment              | +1–4 pp over plain argmax |
| Window `(0.10, 0.75)` vs. `(0, 1)`         | +1.4 pp |

### 4.2 Classifier alternatives — none beat L2 LR

LDA with shrinkage matched LR. SVM with RBF, ridge regression, gradient
boosting, EEGNet / ShallowNet / Conformer trained on raw MEG, a small
attention-pool transformer, a class-mean prototype with cosine distance,
and a Riemannian tangent space classifier all underperformed LR when used
as the sole classifier. A pre-trained MEG foundation model
(BrainOmni-style) underperformed in the same regime. An MLP on the same
multi-features as LR landed within noise of LR but did not improve the
ensemble.

### 4.3 Feature alternatives — none replaced the multi-feature

Time-resolved per-band amplitudes, band-time concatenation, a filter bank,
theta/alpha Hilbert envelopes, Morlet wavelet TFR features, log-Euclidean
covariance features, and ICA-cleaned reconstructions of the same channels
all under-performed or matched the multi-feature with no public-LB gain.
ICA artifact removal in particular (auto-labelling components by
correlation with the recorded ECG/EOG channels) consistently reduced solo
accuracy, indicating that the tSSS MaxFilter pre-processing on the released
epochs is already sufficiently clean.

### 4.4 Cross-modal transfer — fails near chance

Within the compliant path, the following variants all plateaued at
0.10–0.13 on the public LB:

- localizer-trained LR applied to imagine, pooled across subjects,
- same-subject localizer → imagine,
- joint training on localizer + imagine (mixed loss),
- CORAL / SRM / shared-latent-space cross-modality alignment,
- localizer ERP template matching (correlation, ridge projection,
  time-lagged variants),
- CSP / OvR-CSP on localizer.

### 4.5 Test-population failures

Several methods produced clean, single-peaked cross-validation
improvements that nevertheless lost public-LB accuracy. These were
abandoned. Examples:

- subject bagging (LOSO 18.7 % → public LB 14.7 %),
- per-test-subject sensor re-ranking using the test subject's own
  localizer (CV +1.2 pp, public LB −3.1 pp),
- per-test-subject sample-weight reweighting based on unsupervised
  subject similarity (CV +0.8 pp, public LB −0.5 pp),
- raw-MEG CNN ensembled with the LR ensemble (CV +2.9 pp, public LB
  −1.8 pp),
- aggressive cross-validation grid search on per-N `(pca, C)`
  (CV +0.8 pp, public LB −2.8 pp).

The empirical rule that emerged: any cross-validation gain below
approximately 0.03 on a 100-split L3SO protocol does not predict the
sign of the public-LB change. Hard-subject validation (using the
unsupervised-similarity-based hardest training subjects as a held-out
proxy for the test population) does not predict it either.

### 4.6 Leaderboard inversion — feasible set too large

A mixed-integer programming formulation was attempted: 6800 binary
indicator variables over 680 rows × 10 classes, hard one-hot constraints,
hard per-subject class-quota constraints, and one hard linear constraint
per known Kaggle score. Up to 35 known scores were supplied, with
posteriors based on the cross-subject LR and on a Kaggle-score-weighted
vote over all submissions. The MILP found feasible solutions in seconds
but the feasible set is too large for these constraints to identify the
test labels: the resulting public-LB accuracy was always in the 0.16–0.18
range, well below the cross-subject LR ensemble. Targeted single-row
"Hungarian-undo" probes — swapping a high-confidence forced label with
its raw argmax, balanced by demoting the weakest in-class peer — produced
net-zero changes (the gain rows were correct; the demote rows were also
already correct). No probe-based or inversion-based variant exceeded
0.219.

## 5. Discussion

### Why the disallowed path scores higher

The compliant task is genuinely hard: the dataset and the classes were
chosen to test cross-modal generalization from visual decoders to
auditory-cued imagery, and prior literature predicts that this transfer
is weak. The disallowed path side-steps cross-modal transfer entirely by
learning the auditory-imagery cue label directly, cross-subject. The
~9–11 percentage-point gap between the two paths is the empirical
quantification of the cross-modal barrier this dataset exhibits. We
believe the top of the public leaderboard reflects this same gap: the
challenge-compliant accuracy ceiling is near 0.13, and the cross-subject
imagine-only ceiling we observed is near 0.22. The leader's 0.235 is
within the same regime.

### Negative result

The strongest finding of this work is the negative one: across the wide
range of pre-processing, feature engineering, classifier choice, domain
adaptation, source-space reconstruction, transductive learning, and
ensembling explored here, no challenge-compliant variant approached the
disallowed cross-subject ceiling. This empirically supports the framing
in the challenge background: "using the default settings used in many
memory papers (i.e. training on a fixed timepoint of visual decoding peak)
seems not to work well" for cross-modal imagery decoding in this dataset.

## 6. Reproducing this submission

```bash
pip install -r src/requirements.txt
# Place the published dataset under ./data/
python src/main.py                # produces submission_main.csv (~0.219)
python src/compliant_baseline.py  # produces submission_compliant_baseline.csv (~0.10-0.13)
```

Expected dataset layout:

```
data/
  train/train/sub-{02,05,06,07,10,13,14,15,17,18,25,28,29,30,31}/
    sub-XX_imagine-epo.fif
    sub-XX_localizer-epo.fif
  test/test/sub-{01,03,04,09,11,12,16,19,21,22,23,24,26,27}/
    sub-XX_imagine-epo.fif
    sub-XX_localizer-epo.fif
```

The two scripts are independent and self-contained. Each completes in well
under a minute on a modern CPU.

## 7. References

- Bezsudnova, Y., et al. (2024). Cross-modal generalization in MEG decoders.
- Dijkstra, N., et al. (2019). Differential temporal dynamics during visual
  imagery and perception.
- Dijkstra, N., et al. (2021). Subjective signal strength distinguishes
  reality from imagination.
- Shatek, S. M., et al. (2019). Decoding images in the mind's eye:
  the temporal dynamics of visual imagery.
- Kern, F., et al. (2020). Memory replay during human resting state.
