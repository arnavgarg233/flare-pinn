#!/usr/bin/env python3
"""
Train Classical Baselines: Random Forest and Logistic Regression

Trains RF and Logistic Regression on handcrafted physics features extracted
from magnetogram windows, evaluates on test set, and generates metrics.

Usage:
    python tools/train_classical_baselines.py --split train --output-prefix train_80
    python tools/train_classical_baselines.py --split test --model-path models.pkl --output-prefix test_15
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pinn.rf_guidance import compute_handcrafted_features


def load_magnetogram_frames(
    harpnum: int,
    t0: pd.Timestamp,
    consolidated_dir: Path,
    input_hours: int = 48,
    target_size: int = 128
) -> np.ndarray | None:
    """Load magnetogram frames for a window."""
    # Look for consolidated bundle (format: H<harpnum>.npz)
    bundle_path = consolidated_dir / f"H{harpnum}.npz"
    
    if not bundle_path.exists():
        return None
    
    try:
        bundle = np.load(bundle_path, allow_pickle=True)
        frames = bundle['frames']  # [N, 3, H, W] where 3=[Bx, By, Bz]
        timestamps = bundle['timestamps']  # [N] timestamps
        
        # Get frames within window
        t_end = t0 + pd.Timedelta(hours=input_hours)
        
        # Convert timestamps to pandas
        ts_array = pd.to_datetime(timestamps)
        
        # Find frames in window
        mask = (ts_array >= t0) & (ts_array < t_end)
        
        if not mask.any():
            return None
        
        # Extract Bz component (index 2) for matching frames
        bz_frames = frames[mask, 2, :, :]  # [T, H, W]
        
        if bz_frames.shape[0] == 0:
            return None
            
        return bz_frames
        
    except Exception as e:
        # print(f"Error loading {bundle_path}: {e}")
        return None


def extract_features_from_windows(
    windows_df: pd.DataFrame,
    consolidated_dir: Path,
    input_hours: int = 48,
    target_size: int = 128,
    verbose: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Extract handcrafted physics features from all windows.
    
    Returns:
        features: [N, n_features] array
        feature_names: list of feature names
    """
    all_features = []
    feature_names = None
    
    iterator = tqdm(windows_df.iterrows(), total=len(windows_df), desc="Extracting features") if verbose else windows_df.iterrows()
    
    for idx, row in iterator:
        # Load magnetogram frames
        frames = load_magnetogram_frames(
            harpnum=row['harpnum'],
            t0=pd.Timestamp(row['t0']),
            consolidated_dir=consolidated_dir,
            input_hours=input_hours,
            target_size=target_size
        )
        
        if frames is None or len(frames) == 0:
            # Use dummy features if no data
            if feature_names is None:
                # Initialize with dummy to get feature names
                dummy = compute_handcrafted_features(np.zeros((1, target_size, target_size)))
                feature_names = list(dummy.keys())
            
            features_dict = {name: 0.0 for name in feature_names}
        else:
            # Extract features
            features_dict = compute_handcrafted_features(frames)
            
            if feature_names is None:
                feature_names = list(features_dict.keys())
        
        # Convert to array
        features_array = np.array([features_dict[name] for name in feature_names])
        all_features.append(features_array)
    
    return np.stack(all_features, axis=0), feature_names


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> dict[str, Any]:
    """Train RandomForest and LogisticRegression."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*70)
    print("Training Classical Baselines")
    print("="*70)
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train_scaled, y_train)
    
    train_acc_rf = rf.score(X_train_scaled, y_train)
    print(f"RF Training Accuracy: {train_acc_rf:.4f}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        penalty='l2',
        C=0.1,
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    lr.fit(X_train_scaled, y_train)
    
    train_acc_lr = lr.score(X_train_scaled, y_train)
    print(f"LR Training Accuracy: {train_acc_lr:.4f}")
    
    return {
        'rf': rf,
        'lr': lr,
        'scaler': scaler
    }


def evaluate_model(
    model: Any,
    scaler: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> dict[str, Any]:
    """Evaluate model and compute TSS."""
    from sklearn.metrics import roc_curve
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = model.decision_function(X_test_scaled)
    
    # Compute TSS for different thresholds
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    tss_values = tpr - fpr
    best_idx = np.argmax(tss_values)
    
    best_tss = tss_values[best_idx]
    best_threshold = thresholds[best_idx]
    
    # Compute confusion matrix at best threshold
    y_pred = (probs >= best_threshold).astype(int)
    
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    # Compute metrics
    pod = tp / (tp + fn + 1e-10)  # Probability of Detection
    far = fp / (tp + fp + 1e-10)  # False Alarm Ratio
    pofd = fp / (fp + tn + 1e-10)  # Probability of False Detection
    
    tss = pod - pofd
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / (hss_den + 1e-10)
    
    csi = tp / (tp + fp + fn + 1e-10)  # Critical Success Index
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n{model_name} Results:")
    print(f"  TSS: {tss:.4f} (threshold={best_threshold:.4f})")
    print(f"  HSS: {hss:.4f}")
    print(f"  POD: {pod:.4f}")
    print(f"  FAR: {far:.4f}")
    print(f"  CSI: {csi:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    
    return {
        'model_name': model_name,
        'tss': tss,
        'hss': hss,
        'pod': pod,
        'far': far,
        'csi': csi,
        'accuracy': accuracy,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn,
        'threshold': best_threshold,
        'probs': probs,
        'y_true': y_test,
        'thresholds': thresholds,
        'tss_values': tss_values
    }


def main():
    parser = argparse.ArgumentParser(description="Train classical baselines")
    parser.add_argument('--train-windows', type=str, default='data/interim/windows_train_80.parquet',
                        help='Training windows parquet')
    parser.add_argument('--test-windows', type=str, default='data/interim/windows_test_15.parquet',
                        help='Test windows parquet')
    parser.add_argument('--consolidated-dir', type=str, default='~/flare_data/consolidated',
                        help='Consolidated magnetogram directory')
    parser.add_argument('--horizon', type=int, default=24,
                        help='Prediction horizon in hours (6, 12, or 24)')
    parser.add_argument('--input-hours', type=int, default=48,
                        help='Input window hours')
    parser.add_argument('--target-size', type=int, default=128,
                        help='Target image size')
    parser.add_argument('--output-dir', type=str, default='final_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Expand paths
    train_path = Path(args.train_windows).expanduser()
    test_path = Path(args.test_windows).expanduser()
    consolidated_dir = Path(args.consolidated_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load windows
    print("Loading windows...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train windows: {len(train_df)}")
    print(f"Test windows: {len(test_df)}")
    
    # Get labels
    label_col = f'y_geq_M_{args.horizon}h'
    y_train = train_df[label_col].values
    y_test = test_df[label_col].values
    
    print(f"\nTrain positives: {y_train.sum()}/{len(y_train)} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"Test positives: {y_test.sum()}/{len(y_test)} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    # Extract features
    print("\n" + "="*70)
    print("Extracting Training Features")
    print("="*70)
    X_train, feature_names = extract_features_from_windows(
        train_df, consolidated_dir, args.input_hours, args.target_size
    )
    
    print("\n" + "="*70)
    print("Extracting Test Features")
    print("="*70)
    X_test, _ = extract_features_from_windows(
        test_df, consolidated_dir, args.input_hours, args.target_size
    )
    
    print(f"\nFeature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    
    # Train models
    models = train_models(X_train, y_train, args.seed)
    
    # Save models
    model_path = output_dir / 'classical_models.pkl'
    print(f"\nSaving models to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'rf': models['rf'],
            'lr': models['lr'],
            'scaler': models['scaler'],
            'feature_names': feature_names,
            'horizon': args.horizon
        }, f)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on Test Set")
    print("="*70)
    
    rf_results = evaluate_model(models['rf'], models['scaler'], X_test, y_test, 'RandomForest')
    lr_results = evaluate_model(models['lr'], models['scaler'], X_test, y_test, 'LogisticRegression')
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {
            'model': 'RandomForest',
            'horizon_hours': args.horizon,
            'tss': rf_results['tss'],
            'hss': rf_results['hss'],
            'pod': rf_results['pod'],
            'far': rf_results['far'],
            'csi': rf_results['csi'],
            'accuracy': rf_results['accuracy'],
            'tp': rf_results['tp'],
            'fn': rf_results['fn'],
            'fp': rf_results['fp'],
            'tn': rf_results['tn'],
            'threshold': rf_results['threshold']
        },
        {
            'model': 'LogisticRegression',
            'horizon_hours': args.horizon,
            'tss': lr_results['tss'],
            'hss': lr_results['hss'],
            'pod': lr_results['pod'],
            'far': lr_results['far'],
            'csi': lr_results['csi'],
            'accuracy': lr_results['accuracy'],
            'tp': lr_results['tp'],
            'fn': lr_results['fn'],
            'fp': lr_results['fp'],
            'tn': lr_results['tn'],
            'threshold': lr_results['threshold']
        }
    ])
    
    metrics_path = output_dir / 'metrics' / f'classical_baselines_80_5_15_{args.horizon}h.csv'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Save validation results (for confusion matrix generation)
    for results, name in [(rf_results, 'RandomForest'), (lr_results, 'LogisticRegression')]:
        val_path = output_dir / 'validation_results' / f'{name}_{args.horizon}h_test.npz'
        val_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            val_path,
            probs=results['probs'].reshape(-1, 1),  # [N, 1]
            labels=results['y_true'].reshape(-1, 1),  # [N, 1]
            horizons=np.array([args.horizon]),
            thresholds=results['thresholds'],
            tss_values=results['tss_values'].reshape(1, -1)  # [1, n_thresholds]
        )
        print(f"Validation results saved to {val_path}")
    
    print("\n" + "="*70)
    print("✅ Classical baselines training complete!")
    print("="*70)


if __name__ == '__main__':
    main()

