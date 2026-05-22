from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
import yaml

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# hybrid2 direction + label smoothing intervals inspired by Pham et al. 2020
# U-Ones + LSR: U(-1) -> uniform(0.55, 0.85)
# U-Zeros + LSR: U(-1) -> uniform(0.00, 0.30)
POSITIVE_UNCERTAIN = {"Atelectasis", "Edema"}
NEGATIVE_UNCERTAIN = {"Cardiomegaly", "Consolidation", "Pleural Effusion"}


def norm_path_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace('\\\\', '/', regex=False).str.replace('\\', '/', regex=False)


def apply_hybrid2_lsr(row, rng):
    out = {}
    for label in LABELS:
        value = row.get(label, np.nan)
        if pd.isna(value):
            out[label] = 0.0
        elif float(value) == 1.0:
            out[label] = 1.0
        elif float(value) == 0.0:
            out[label] = 0.0
        elif float(value) == -1.0:
            if label in POSITIVE_UNCERTAIN:
                out[label] = float(rng.uniform(0.55, 0.85))
            else:
                out[label] = float(rng.uniform(0.00, 0.30))
        else:
            # Fallback for unexpected values
            out[label] = float(value)
    return pd.Series(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--base-strategy', default='kaggle80k_hybrid2_frontal')
    parser.add_argument('--out-strategy', default='kaggle80k_hybrid2_lsr_frontal')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(cfg['project_dir'])
    archive_root = Path(cfg['archive_root'])
    train_csv = Path(cfg['train_csv'])
    local_ready = project_dir / 'local_ready'
    reports_dir = project_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_train_path = local_ready / f'train_{args.base_strategy}_local.csv'
    base_val_path = local_ready / f'val_{args.base_strategy}_local.csv'
    base_test_path = local_ready / f'test_{args.base_strategy}_local.csv'
    base_official_path = local_ready / f'official_valid_{args.base_strategy}_local.csv'

    out_train_path = local_ready / f'train_{args.out_strategy}_local.csv'
    out_val_path = local_ready / f'val_{args.out_strategy}_local.csv'
    out_test_path = local_ready / f'test_{args.out_strategy}_local.csv'
    out_official_path = local_ready / f'official_valid_{args.out_strategy}_local.csv'

    for p in [base_train_path, base_val_path, base_test_path, base_official_path, train_csv]:
        if not p.exists():
            raise FileNotFoundError(f'Missing required file: {p}')

    print('[LOAD] Base hard-label train split:', base_train_path)
    base_train = pd.read_csv(base_train_path)
    print('  rows:', len(base_train))

    print('[LOAD] Raw Kaggle train.csv:', train_csv)
    raw = pd.read_csv(train_csv)

    if 'Path' not in raw.columns:
        raise ValueError('Raw train.csv must contain a Path column.')

    raw['_path_key'] = norm_path_series(raw['Path'])

    # Determine key column in base train split
    if 'Path' in base_train.columns:
        base_train['_path_key'] = norm_path_series(base_train['Path'])
    elif 'path' in base_train.columns:
        base_train['_path_key'] = norm_path_series(base_train['path'])
    elif 'image_path' in base_train.columns:
        base_train['_path_key'] = norm_path_series(base_train['image_path'])
    elif 'local_image_path' in base_train.columns:
        # local_image_path is absolute; recover the CheXpert relative section after archive root if possible
        local_norm = norm_path_series(base_train['local_image_path'])
        archive_norm = str(archive_root).replace('\\', '/')
        base_train['_path_key'] = local_norm.str.replace(archive_norm + '/', '', regex=False)
    else:
        raise ValueError('Could not find a path column in base train split.')

    # Keep only needed raw columns and merge to recover -1 uncertainty labels
    raw_small = raw[['_path_key'] + LABELS].copy()
    raw_small = raw_small.rename(columns={label: f'raw_{label}' for label in LABELS})

    merged = base_train.merge(raw_small, on='_path_key', how='left')
    missing_raw = merged[[f'raw_{label}' for label in LABELS]].isna().all(axis=1).sum()
    print('[MERGE] rows with no raw label match:', int(missing_raw))
    if missing_raw > 0:
        print('[WARN] Some rows did not match raw train.csv. They will keep the hard labels from the base split.')

    rng = np.random.default_rng(args.seed)

    # Apply soft labels for training only. Use raw labels when available; otherwise keep base hard labels.
    raw_label_cols = [f'raw_{label}' for label in LABELS]
    soft_df = merged.copy()

    raw_rows = soft_df[raw_label_cols].rename(columns={f'raw_{label}': label for label in LABELS})
    soft_labels = raw_rows.apply(lambda r: apply_hybrid2_lsr(r, rng), axis=1)

    for label in LABELS:
        # If raw label missing, keep original hard label
        mask_missing = soft_df[f'raw_{label}'].isna()
        soft_df[label] = soft_labels[label]
        soft_df.loc[mask_missing, label] = base_train.loc[mask_missing, label].astype(float)

    # Remove helper/raw columns
    drop_cols = ['_path_key'] + raw_label_cols
    soft_df = soft_df.drop(columns=[c for c in drop_cols if c in soft_df.columns])

    soft_df.to_csv(out_train_path, index=False)
    shutil.copy2(base_val_path, out_val_path)
    shutil.copy2(base_test_path, out_test_path)
    shutil.copy2(base_official_path, out_official_path)

    print('\n[DONE] Saved LSR train split:')
    print(out_train_path)
    print('[DONE] Copied hard-label val/test/official_valid splits:')
    print(out_val_path)
    print(out_test_path)
    print(out_official_path)

    print('\n[TRAIN LABEL SUMMARY - soft means]')
    for label in LABELS:
        vals = soft_df[label].astype(float)
        print(f'{label:18s} mean={vals.mean():.4f} min={vals.min():.3f} max={vals.max():.3f}')

    summary = {
        'base_strategy': args.base_strategy,
        'out_strategy': args.out_strategy,
        'seed': args.seed,
        'train_rows': int(len(soft_df)),
        'missing_raw_matches': int(missing_raw),
        'lsr_positive_uncertain_interval': [0.55, 0.85],
        'lsr_negative_uncertain_interval': [0.0, 0.30],
        'positive_uncertain_labels': sorted(list(POSITIVE_UNCERTAIN)),
        'negative_uncertain_labels': sorted(list(NEGATIVE_UNCERTAIN)),
    }
    pd.Series(summary).to_json(reports_dir / f'lsr_summary_{args.out_strategy}.json', indent=2)
    print('[DONE] Saved report:', reports_dir / f'lsr_summary_{args.out_strategy}.json')


if __name__ == '__main__':
    main()
