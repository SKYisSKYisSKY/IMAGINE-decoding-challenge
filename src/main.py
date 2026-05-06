"""
main.py - Cross-subject imagine-trial decoder.

NOTE on compliance: this script implements the path the challenge instructions
explicitly designate as "not allowed for solving the challenge" (overview,
line 53). The same line acknowledges "we'll be equally impressed if you
achieve this." For a challenge-compliant baseline (localizer-trained,
Kaggle ~0.10-0.13), see compliant_baseline.py.

Pipeline summary:
  - Load imagine epochs, gradiometers only (204 channels)
  - Crop to (0.10, 0.75) seconds after word onset
  - Per-subject channel-wise z-score normalization
  - Pooled ANOVA F-score sensor ranking on training imagine labels,
    aggregated within Vectorview gradiometer pairs
  - For each N in {1, 2, 3, 4, 5, 7, 10}:
      - Top-N sensor pairs (= 2N grad channels)
      - Multi-feature: downsampled time + per-channel mean/std/max + 5 PSD bands
      - StandardScaler + optional PCA + L2 Logistic Regression
  - Average predicted probabilities across the 7 N values
  - Per-test-subject Hungarian assignment with quota = n_trials // 10
"""
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from scipy.signal import welch
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train' / 'train'
TEST_DIR = DATA_DIR / 'test' / 'test'
OUTPUT_PATH = Path('submission_main.csv')

N_CLASSES = 10
SFREQ = 100.0
SEED = 42

LABELS = ['apple', 'bicycle', 'brush', 'cake', 'clown',
          'cup', 'desk', 'foot', 'mountain', 'zebra']  # alphabetical, matches event_id 1..10
LABEL_MAP = {i + 1: name for i, name in enumerate(LABELS)}

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}

BEST_WIN = (0.10, 0.75)

# Top-N sensor configurations for the ensemble.
Ns = [1, 2, 3, 4, 5, 7, 10]

# Per-N (PCA variance fraction, LR C) configuration, hand-tuned.
PER_N_CFG = {
    1:  (None, 0.10),
    2:  (0.97, 0.08),
    3:  (None, 0.05),
    4:  (0.97, 0.03),
    5:  (0.97, 0.10),
    7:  (0.97, 0.08),
    10: (None, 0.05),
}


# ============================================================
# Data loading
# ============================================================
def load_epochs(base_dir, sub_id, task='imagine', ch_type='grad'):
    fp = base_dir / sub_id / f'{sub_id}_{task}-epo.fif'
    ep = mne.read_epochs(str(fp), preload=True, verbose=False)
    ep.pick(ch_type)
    return ep.get_data(), ep.events[:, 2], ep.ch_names


def time_crop(X, tmin, tmax, epoch_tmin=-0.2):
    i0 = max(0, int(round((tmin - epoch_tmin) * SFREQ)))
    i1 = min(X.shape[2], int(round((tmax - epoch_tmin) * SFREQ)))
    return X[:, :, i0:i1]


def subject_normalize(X):
    """Per-subject channel-wise z-score across epochs and time."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mu) / std


# ============================================================
# Feature extraction
# ============================================================
def extract_multi_features(X, sfreq=SFREQ):
    """Concatenate downsampled time series, per-channel summary stats,
    and per-band PSD averages."""
    n_ep, n_ch, n_t = X.shape
    feats = []
    step = max(1, n_t // 50)
    feats.append(X[:, :, ::step].reshape(n_ep, -1))
    feats.append(X.mean(axis=2))
    feats.append(X.std(axis=2))
    feats.append(np.abs(X).max(axis=2))
    nperseg = min(n_t, max(16, n_t // 2))
    freqs, psd = welch(X, fs=sfreq, axis=-1, nperseg=nperseg)
    for _, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() > 0:
            feats.append(psd[:, :, mask].mean(axis=-1))
    return np.hstack(feats)


# ============================================================
# Pooled ANOVA sensor ranking
# ============================================================
def compute_anova_sensor_ranking(sn_data, labels_d, train_subs, ch_names):
    """Rank Vectorview gradiometer pairs by the sum of per-channel ANOVA
    F-scores against pooled training imagine labels."""
    pooled_X = np.concatenate([sn_data[s] for s in train_subs])
    pooled_y = np.concatenate([labels_d[s] for s in train_subs])
    n_ch = pooled_X.shape[1]
    ch_scores = np.zeros(n_ch)
    for ch in range(n_ch):
        x_ch = pooled_X[:, ch, :]
        feat_ch = np.hstack([x_ch,
                             x_ch.mean(axis=1, keepdims=True),
                             x_ch.std(axis=1, keepdims=True)])
        F, _ = f_classif(feat_ch, pooled_y)
        ch_scores[ch] = np.nanmean(F)
    sensor_groups = {}
    for i, c in enumerate(ch_names):
        sensor_groups.setdefault(c[3:6], []).append(i)
    sensor_scores = [(s, idxs, ch_scores[idxs].sum())
                     for s, idxs in sensor_groups.items()]
    sensor_scores.sort(key=lambda x: -x[2])
    return sensor_scores


# ============================================================
# Classifier and balanced assignment
# ============================================================
def fit_predict(F_tr, y_tr, F_te, pca_var, C, seed=SEED):
    """Standardize, optional PCA, fit LR on the train fold, return test
    class probabilities."""
    sc = StandardScaler().fit(F_tr)
    Ftr = sc.transform(F_tr)
    Fte = sc.transform(F_te)
    if pca_var is not None:
        pca = PCA(n_components=pca_var, svd_solver='full',
                  random_state=seed).fit(Ftr)
        Ftr = pca.transform(Ftr)
        Fte = pca.transform(Fte)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit(Ftr, y_tr)
    return clf.predict_proba(Fte)


def hungarian_balanced(prob_matrix, n_per_class):
    """Per-subject balanced label assignment.

    Each subject's trials are assigned to classes such that exactly
    `n_per_class` trials go to each of the 10 classes. Total assignment
    cost (sum of -log(prob)) is minimized.
    """
    n_trials, n_classes = prob_matrix.shape
    cost = np.zeros((n_trials, n_trials))
    for c in range(n_classes):
        log_p = -np.log(np.clip(prob_matrix[:, c], 1e-10, 1.0))
        for j in range(c * n_per_class, (c + 1) * n_per_class):
            cost[:, j] = log_p
    _, col_idx = linear_sum_assignment(cost)
    labels = np.zeros(n_trials, dtype=int)
    for r, c in zip(range(n_trials), col_idx):
        labels[r] = (c // n_per_class) + 1
    return labels


# ============================================================
# Main
# ============================================================
def main():
    t0 = _time.time()
    print("Cross-subject imagine-trial decoder")

    train_subs = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    test_subs = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    all_subs = train_subs + test_subs

    print(f"Loading imagine (grad) for {len(all_subs)} subjects...")
    raw_data, labels_d = {}, {}
    ch_names = None
    for base, subs in [(TRAIN_DIR, train_subs), (TEST_DIR, test_subs)]:
        for sub in subs:
            X, y, cn = load_epochs(base, sub, 'imagine', 'grad')
            raw_data[sub] = X
            labels_d[sub] = y
            if ch_names is None:
                ch_names = cn
    print(f"  done [{_time.time() - t0:.0f}s]")

    tmin, tmax = BEST_WIN
    n_t = min(time_crop(raw_data[s], tmin, tmax).shape[2] for s in all_subs)
    sn = {s: subject_normalize(time_crop(raw_data[s], tmin, tmax)[:, :, :n_t])
          for s in all_subs}

    anova = compute_anova_sensor_ranking(sn, labels_d, train_subs, ch_names)
    print(f"  Top-10 sensor pairs: {[s for s, _, _ in anova[:10]]}")

    print(f"\nBuilding multi-features for Ns={Ns}...")
    feats = {}
    for N in Ns:
        idx = [j for s, idxs, _ in anova[:N] for j in idxs]
        feats[N] = {s: extract_multi_features(sn[s][:, idx, :])
                    for s in all_subs}

    print(f"\nTraining 7 ensemble members and predicting test...")
    test_probs_avg = {sub: np.zeros((len(raw_data[sub]), N_CLASSES))
                      for sub in test_subs}
    for N in Ns:
        pca_var, C = PER_N_CFG[N]
        F_tr = np.concatenate([feats[N][s] for s in train_subs])
        y_tr = np.concatenate([labels_d[s] for s in train_subs])
        for sub in test_subs:
            test_probs_avg[sub] += fit_predict(
                F_tr, y_tr, feats[N][sub], pca_var, C)
    for sub in test_subs:
        test_probs_avg[sub] /= len(Ns)

    predictions = {'ID': [], 'label': []}
    for sub in test_subs:
        n_per_class = len(test_probs_avg[sub]) // N_CLASSES
        preds = hungarian_balanced(test_probs_avg[sub], n_per_class)
        for i, lab in enumerate(preds):
            predictions['ID'].append(f"{sub}_{i + 1}")
            predictions['label'].append(LABEL_MAP[lab])

    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(df)} rows)")
    print(f"Total: {_time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
