# IMAGINE Decoding Challenge

Code and submissions for the [IMAGINE decoding challenge](https://github.com/skjerns/imagine-decoding-challenge).
The repository contains two self-contained pipelines for the test imagine
trials:

- **`src/main.py`** — cross-subject imagine-trial decoder. Public LB 0.219.
- **`src/compliant_baseline.py`** — challenge-compliant baseline. Public LB
  approximately 0.10–0.13 (near chance).

> **Compliance disclosure.** The main pipeline trains its classifier on the
> imagine train-set labels and decodes across participants without using
> localizer data. Per overview line 53, this path is "not allowed for
> solving the challenge"; the same line acknowledges "we'll be equally
> impressed if you achieve this." The compliant baseline is the official
> task as stated. Both submissions are released for transparency.
> Detailed discussion: `WRITEUP_EN.md` / `WRITEUP_CN.md`.

## Quick start

```bash
pip install -r src/requirements.txt
# Place the published dataset under ./data/
python src/main.py                # produces submission_main.csv
python src/compliant_baseline.py  # produces submission_compliant_baseline.csv
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

Both scripts complete in well under a minute on a modern CPU.

## Repository contents

```
README.md                                 this file
WRITEUP_EN.md                             full method write-up (English)
WRITEUP_CN.md                             full method write-up (Chinese)
src/
  main.py                                 cross-subject imagine decoder
  compliant_baseline.py                   localizer-trained baseline
  requirements.txt                        minimal Python dependencies
submissions/
  submission_main.csv                     final submission, public LB 0.219
  submission_compliant_baseline.csv       baseline submission, public LB ~0.10-0.13
```

## Method (one paragraph)

`main.py` uses gradiometers only, time-cropped to (0.10, 0.75) s after
imagine word onset and per-subject channel-wise z-scored. A pooled ANOVA
F-score on the imagine training labels, aggregated within Vectorview
gradiometer pairs, selects roughly 14 of 204 channels (bilateral
temporo-parietal). Seven Logistic Regression ensemble members, each at a
different top-N sensor-pair configuration `N ∈ {1, 2, 3, 4, 5, 7, 10}`,
operate on a multi-feature representation (downsampled time series +
per-channel mean/std/max + Welch PSD per band). Their predicted
probabilities are averaged. Per-test-subject Hungarian assignment enforces
the uniform 5-per-class (3-per-class for the 30-trial subject) prior.
Public LB 0.219.

`compliant_baseline.py` uses the same feature extraction and Hungarian
assignment, but the classifier is trained exclusively on the **localizer**
epochs of all 29 subjects with **localizer** labels; imagine train labels
are never seen by the classifier. Public LB ~0.10–0.13.

## What did not work

Briefly: cross-modal transfer (visual localizer → imagery) plateaus at
chance; late imagine windows are at chance; template-MRI source-space
reconstruction collapses to chance; transductive whitening, ICA artifact
removal, sub-window stacking, sample bagging, raw-MEG CNN ensembled with
the LR ensemble, and MILP-based leaderboard inversion with up to 35 known
scores all reduce public-LB accuracy. Full audit trail in
`WRITEUP_EN.md` / `WRITEUP_CN.md`, Section 4.

## License

Code released under MIT.
