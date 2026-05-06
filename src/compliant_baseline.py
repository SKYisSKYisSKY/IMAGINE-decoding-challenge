"""
compliant_baseline.py - Challenge-compliant baseline.

Per overview line 53, the official task is to "train a classifier on the data
after stimulus onset of the localizer" and apply it to imagine epochs. This
baseline does exactly that:

  1. Train a Logistic Regression on LOCALIZER epochs (visual stimulus onset)
     using LOCALIZER labels, pooled across all 29 subjects.
  2. Apply the trained classifier to IMAGINE epochs (auditory-cued imagery).
  3. No imagine train labels used for classifier fitting.

Expected Kaggle ≈ 0.10-0.13 (near chance). This empirically demonstrates
that visual perception decoders DO NOT cross-modally generalize to imagery
in this dataset under standard MEG decoding pipelines, consistent with
Bezsudnova et al. (2024) and Dijkstra et al. (2019, 2021).

This file is provided as a complement to main_v31.py for transparent
disclosure: the high-scoring v31 method is explicitly the path the
instructions designate as "not allowed for solving the challenge", and the
compliant approach plateaus near chance.
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


DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train' / 'train'
TEST_DIR = DATA_DIR / 'test' / 'test'
OUTPUT_PATH = Path('submission_compliant_baseline.csv')

N_CLASSES = 10
SFREQ = 100.0
SEED = 42

LABELS = ['apple', 'bicycle', 'brush', 'cake', 'clown',
          'cup', 'desk', 'foot', 'mountain', 'zebra']
LABEL_MAP = {i + 1: name for i, name in enumerate(LABELS)}

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}

# Localizer visual peak window (~100-400 ms after image onset)
LOC_WIN = (0.1, 0.4)
# Imagine application window: same span post-word-onset for fair comparison
IMG_WIN = (0.1, 0.4)

# v31-style ANOVA top-N grad-pair sensor selection (used here ONLY on
# localizer data — train labels for ranking are localizer labels, not imagine)
TOP_N = 10


def load_epochs(base, sub, task, ch_type='grad'):
    fp = base / sub / f'{sub}_{task}-epo.fif'
    ep = mne.read_epochs(str(fp), preload=True, verbose=False)
    ep.pick(ch_type)
    return ep.get_data(), ep.events[:, 2], ep.ch_names


def time_crop(X, tmin, tmax, epoch_tmin=-0.2):
    i0 = max(0, int(round((tmin - epoch_tmin) * SFREQ)))
    i1 = min(X.shape[2], int(round((tmax - epoch_tmin) * SFREQ)))
    return X[:, :, i0:i1]


def subject_normalize(X):
    mu = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mu) / std


def extract_multi_features(X, sfreq=SFREQ):
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


def hungarian(prob_matrix, n_per_class):
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


def compute_anova_sensor_ranking(sn_data, labels_d, subjects, ch_names):
    pooled_X = np.concatenate([sn_data[s] for s in subjects])
    pooled_y = np.concatenate([labels_d[s] for s in subjects])
    n_ch = pooled_X.shape[1]
    ch_scores = np.zeros(n_ch)
    for ch in range(n_ch):
        x = pooled_X[:, ch, :]
        feat = np.hstack([x, x.mean(axis=1, keepdims=True),
                          x.std(axis=1, keepdims=True)])
        F, _ = f_classif(feat, pooled_y)
        ch_scores[ch] = np.nanmean(F)
    groups = {}
    for i, c in enumerate(ch_names):
        groups.setdefault(c[3:6], []).append(i)
    out = [(s, idxs, ch_scores[idxs].sum()) for s, idxs in groups.items()]
    out.sort(key=lambda x: -x[2])
    return out


def main():
    t0 = _time.time()
    print("Compliant baseline: train on LOCALIZER, apply to IMAGINE")

    train_subs = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    test_subs = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    all_subs = train_subs + test_subs

    print("Loading localizer + imagine (grad)...")
    loc_data, loc_y = {}, {}
    img_data = {}
    ch_names = None
    for base, subs in [(TRAIN_DIR, train_subs), (TEST_DIR, test_subs)]:
        for sub in subs:
            Xl, yl, cn = load_epochs(base, sub, 'localizer', 'grad')
            Xi, _, _ = load_epochs(base, sub, 'imagine', 'grad')
            loc_data[sub] = Xl
            loc_y[sub] = yl
            img_data[sub] = Xi
            if ch_names is None:
                ch_names = cn
    print(f"  loaded  [{_time.time()-t0:.0f}s]")

    # Crop and normalize localizer
    n_tl = min(time_crop(loc_data[s], *LOC_WIN).shape[2] for s in all_subs)
    sn_loc = {s: subject_normalize(time_crop(loc_data[s], *LOC_WIN)[:, :, :n_tl])
              for s in all_subs}
    # Crop and normalize imagine to same span
    n_ti = min(time_crop(img_data[s], *IMG_WIN).shape[2] for s in all_subs)
    sn_img = {s: subject_normalize(time_crop(img_data[s], *IMG_WIN)[:, :, :n_ti])
              for s in all_subs}

    # Use the SAME number of time samples (truncate to min)
    n_t = min(n_tl, n_ti)
    sn_loc = {s: sn_loc[s][:, :, :n_t] for s in all_subs}
    sn_img = {s: sn_img[s][:, :, :n_t] for s in all_subs}

    # ANOVA ranking on LOCALIZER labels (allowed - localizer is the supervised
    # source per challenge instructions). All 29 subjects' localizer is
    # labeled; using all of them is allowed.
    print(f"\nANOVA sensor ranking on LOCALIZER labels (all 29 subjects)...")
    anova = compute_anova_sensor_ranking(sn_loc, loc_y, all_subs, ch_names)
    top_idx = [j for s, idxs, _ in anova[:TOP_N] for j in idxs]
    print(f"  Top-{TOP_N} sensor pairs: {[s for s, _, _ in anova[:TOP_N]]}")

    # Build features: localizer features for training, imagine features for prediction
    print(f"\nBuilding features...")
    feat_loc = {s: extract_multi_features(sn_loc[s][:, top_idx, :]) for s in all_subs}
    feat_img = {s: extract_multi_features(sn_img[s][:, top_idx, :]) for s in test_subs}

    # Train LR on localizer of ALL subjects (allowed: localizer labels are public)
    F_tr = np.concatenate([feat_loc[s] for s in all_subs])
    y_tr = np.concatenate([loc_y[s] for s in all_subs])
    print(f"  Train set: {F_tr.shape[0]} localizer trials")

    sc = StandardScaler().fit(F_tr)
    Ftr = sc.transform(F_tr)
    pca = PCA(n_components=0.95, svd_solver='full', random_state=SEED).fit(Ftr)
    Ftr = pca.transform(Ftr)
    print(f"  PCA dim: {Ftr.shape[1]}")

    clf = LogisticRegression(C=0.10, max_iter=5000, random_state=SEED)
    clf.fit(Ftr, y_tr)
    print(f"  Train accuracy on localizer: "
         f"{clf.score(Ftr, y_tr):.3f}")

    # Apply to imagine test
    predictions = {'ID': [], 'label': []}
    for sub in test_subs:
        Fte = pca.transform(sc.transform(feat_img[sub]))
        probs = clf.predict_proba(Fte)
        n_per_class = len(probs) // N_CLASSES
        preds = hungarian(probs, n_per_class)
        for i, lab in enumerate(preds):
            predictions['ID'].append(f"{sub}_{i+1}")
            predictions['label'].append(LABEL_MAP[lab])

    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(df)} rows)")
    print(f"\nNote: this baseline trains ONLY on localizer; imagine train")
    print(f"labels are NEVER used for classifier supervision. Expected Kaggle")
    print(f"score ≈ 0.10-0.13 (near chance), demonstrating the cross-modality")
    print(f"barrier the challenge investigates.")
    print(f"\nTotal: {_time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
