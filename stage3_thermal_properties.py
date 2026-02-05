#!/usr/bin/env python3
"""
Streamlined Thermal Regression Model with Essential Visualizations Only
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5
EPOCHS = 600
BATCH_SIZE = 64
MAX_LR = 1.5e-3
WEIGHT_DECAY = 2e-3
PATIENCE = 60
RANDOM_SEED = 42
TARGET_NAMES = ['Tg', 'Tx', 'Tl']
TOP_FEATURES = 100
PHYSICS_ALPHA = 4.0
MIN_MARGIN = 25.0

OUTPUT_DIR = "STAGE3_THERMAL_REGRESSOR_STREAMLINED"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURIZED_CSV = "featurized_metallic_glass_stage1.csv"
STAGE1_MODEL = os.path.join(OUTPUT_DIR, "stage1_regression_model.pth")
STAGE1_TOP = os.path.join(OUTPUT_DIR, "stage1_top_features.npy")
STAGE1_SCALER = os.path.join(OUTPUT_DIR, "stage1_scaler.joblib")
PARITY_PLOT = os.path.join(OUTPUT_DIR, "parity_plots_publication.png")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ---------------- Model Architecture ----------------
class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.attn(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.BatchNorm1d(dim * 2),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return F.silu(x + self.block(x))

class GlassThermalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attention = FeatureAttention(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, 0.2),
            ResidualBlock(hidden_dim, 0.15),
            ResidualBlock(hidden_dim, 0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, x):
        x = self.attention(x)
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.head(x)
    
    def save_model_architecture(self, filepath):
        """Save the complete model architecture for later reconstruction"""
        model_info = {
            'input_dim': self.head[2].in_features * 2,
            'hidden_dim': self.input_layer[0].out_features,
            'state_dict': self.state_dict(),
            'class_name': self.__class__.__name__
        }
        torch.save(model_info, filepath)
        print(f"Model architecture saved to {filepath}")

# ---------------- Physics-Informed Loss ----------------
class PhysicsLoss(nn.Module):
    def __init__(self, alpha=PHYSICS_ALPHA):
        super().__init__()
        self.mae = nn.L1Loss()
        self.alpha = alpha
    def forward(self, pred, true):
        mae_loss = self.mae(pred, true)
        penalty1 = F.relu(pred[:, 0] - pred[:, 1] + MIN_MARGIN).mean()
        penalty2 = F.relu(pred[:, 1] - pred[:, 2] + MIN_MARGIN).mean()
        return mae_loss + self.alpha * (penalty1 + penalty2)

# ---------------- Essential Visualization ----------------
def plot_parity_publication(y_true, y_pred, targets=['Tg', 'Tx', 'Tl'], save_path=None):
    """Generates professional parity plots with metrics."""
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), dpi=300)
    colors = ['#08519c', '#a50f15', '#006d2c']

    for i, target in enumerate(targets):
        ax = axes[i]
        t, p = y_true[:, i], y_pred[:, i]
        
        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        
        # Regression Line and Scatter
        sns.regplot(x=t, y=p, ax=ax, scatter_kws={'alpha':0.4, 's':60, 'color':colors[i]}, 
                    line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2.5})
        
        # 45-degree Identity Line
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, color='darkgray', lw=2, linestyle='-', zorder=1)

        # Formatting
        ax.set_title(target, fontsize=28, fontweight='bold', pad=20)
        ax.set_xlabel(f'Experimental {target} (K)', fontsize=22, fontweight='bold')
        ax.set_ylabel(f'Predicted {target} (K)', fontsize=22, fontweight='bold')
        
        # Bold Stats Box
        stats_text = f"$R^2 = {r2:.3f}$\n$MAE = {mae:.2f}$ K\n$RMSE = {rmse:.2f}$ K"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

        ax.tick_params(axis='both', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n[INFO] Publication-quality parity plot saved as '{save_path}'")
    else:
        plt.savefig(PARITY_PLOT, bbox_inches='tight')
        print(f"\n[INFO] Publication-quality parity plot saved as '{PARITY_PLOT}'")
    plt.show()

def plot_residuals_summary(y_true, y_pred, targets=['Tg', 'Tx', 'Tl'], save_path=None):
    """Create a summary residual plot for all targets together"""
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    colors = ['#08519c', '#a50f15', '#006d2c']
    
    for i, target in enumerate(targets):
        ax = axes[i]
        t, p = y_true[:, i], y_pred[:, i]
        residuals = t - p
        
        # Scatter plot
        ax.scatter(p, residuals, alpha=0.6, s=40, color=colors[i])
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(p, residuals, 1)
        p_line = np.poly1d(z)
        ax.plot(p, p_line(p), "r--", alpha=0.8, linewidth=2)
        
        # Calculate metrics
        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)
        
        # Formatting
        ax.set_title(f'{target} Residuals', fontsize=18, fontweight='bold')
        ax.set_xlabel(f'Predicted {target} (K)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Residuals (K)', fontsize=14, fontweight='bold')
        
        # Add metrics text
        ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$\n$MAE = {mae:.2f}$ K', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
        
        ax.tick_params(axis='both', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Residual summary plot saved as '{save_path}'")
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, "residuals_summary.png"), bbox_inches='tight')
        print(f"[INFO] Residual summary plot saved as '{OUTPUT_DIR}/residuals_summary.png'")
    plt.show()

# ---------------- Training Function ----------------
def train_one_fold(model, train_loader, val_loader, fold_idx):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = PhysicsLoss()
    best_mae = float('inf')
    patience_cnt = 0
    best_state, best_preds, best_trues = None, None, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(DEVICE)).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_mae = mean_absolute_error(trues, preds)

        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_trues = preds, trues
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0 or epoch <= 5:
            print(f"Fold {fold_idx} | Epoch {epoch:3d} | Val MAE: {val_mae:.2f} K")

        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_mae, best_preds, best_trues

# ---------------- Save Training Results ----------------
def save_training_results(best_model, best_preds, best_trues, scaler, top_features, 
                         best_global_mae, training_info):
    """Save all important model elements and training results"""
    
    # 1. Save the complete model (architecture + weights)
    complete_model_path = os.path.join(OUTPUT_DIR, "complete_model.pth")
    best_model.save_model_architecture(complete_model_path)
    
    # 2. Save model weights only
    torch.save(best_model.state_dict(), STAGE1_MODEL)
    
    # 3. Save predictions and true values for analysis
    predictions_data = {
        'predictions': best_preds,
        'true_values': best_trues,
        'target_names': TARGET_NAMES
    }
    np.save(os.path.join(OUTPUT_DIR, "predictions_true_values.npy"), predictions_data)
    
    # 4. Save training configuration and results
    config_results = {
        'best_global_mae': best_global_mae,
        'random_seed': RANDOM_SEED,
        'k_folds': K_FOLDS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'max_lr': MAX_LR,
        'weight_decay': WEIGHT_DECAY,
        'patience': PATIENCE,
        'top_features': TOP_FEATURES,
        'physics_alpha': PHYSICS_ALPHA,
        'min_margin': MIN_MARGIN,
        'device': str(DEVICE),
        'training_samples': training_info.get('training_samples', 0),
        'total_samples': training_info.get('total_samples', 0)
    }
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    config_path = os.path.join(OUTPUT_DIR, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_results, f, indent=4, default=convert_numpy)
    
    print(f"\n[SAVED] All model elements saved to '{OUTPUT_DIR}' folder:")
    print(f"  - Complete model: {complete_model_path}")
    print(f"  - Model weights: {STAGE1_MODEL}")
    print(f"  - Top features: {STAGE1_TOP}")
    print(f"  - Scaler: {STAGE1_SCALER}")
    print(f"  - Parity plot: {PARITY_PLOT}")
    print(f"  - Residuals summary: {os.path.join(OUTPUT_DIR, 'residuals_summary.png')}")
    print(f"  - Predictions data: {os.path.join(OUTPUT_DIR, 'predictions_true_values.npy')}")
    print(f"  - Training config: {config_path}")

# ---------------- Prediction Functions ----------------
def predict_on_full_dataset():
    """Make predictions on the entire dataset using the trained model"""
    print("\n=== Making predictions on the entire dataset ===")
    
    # Check if model exists
    if not os.path.exists(STAGE1_MODEL):
        raise FileNotFoundError(f"Model not found at {STAGE1_MODEL}. Train the model first.")
    
    # Load the full dataset
    df = pd.read_csv(FEATURIZED_CSV)
    
    # Load the trained model components
    top_features = np.load(STAGE1_TOP)
    scaler = joblib.load(STAGE1_SCALER)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in TARGET_NAMES and np.issubdtype(df[c].dtype, np.number)]
    
    # Make sure all top features exist in the dataset
    missing_features = [f for f in top_features if f not in feature_cols]
    if missing_features:
        print(f"Warning: {len(missing_features)} features are missing from the dataset")
        print("Available features in dataset:", len(feature_cols))
        print("Top features from training:", len(top_features))
        # Only use features that exist in both
        common_features = [f for f in top_features if f in feature_cols]
        print(f"Using {len(common_features)} common features")
        top_features = np.array(common_features)
    
    # Prepare features
    X_full = df[top_features].fillna(0).values
    X_scaled = scaler.transform(X_full)
    
    # Load model
    model = GlassThermalModel(input_dim=len(top_features))
    model.load_state_dict(torch.load(STAGE1_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Make predictions in batches
    predictions = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(DEVICE)
            batch_pred = model(batch).cpu().numpy()
            predictions.append(batch_pred)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Add predictions to dataframe
    for i, target in enumerate(TARGET_NAMES):
        df[f'Predicted_{target}'] = predictions[:, i]
    
    # Save predictions
    output_csv = os.path.join(OUTPUT_DIR, "all_predictions.csv")
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # Calculate metrics only for samples with non-NaN actual values
    print("\n=== Prediction Summary ===")
    print(f"Total samples predicted: {len(df)}")
    print("\nPrediction vs Actual comparison (only for samples with actual values):")
    
    # Store valid predictions and true values for visualization
    valid_indices = df[TARGET_NAMES].notna().all(axis=1)
    valid_df = df[valid_indices].copy()
    
    if len(valid_df) > 0:
        valid_predictions = valid_df[[f'Predicted_{target}' for target in TARGET_NAMES]].values
        valid_true_values = valid_df[TARGET_NAMES].values
        
        # Generate visualization plots
        print("\nGenerating prediction visualization plots...")
        plot_parity_publication(valid_true_values, valid_predictions, 
                               save_path=os.path.join(OUTPUT_DIR, "prediction_parity.png"))
        plot_residuals_summary(valid_true_values, valid_predictions,
                              save_path=os.path.join(OUTPUT_DIR, "prediction_residuals.png"))
        
        # Print metrics for each target
        for i, target in enumerate(TARGET_NAMES):
            mask = df[target].notna()
            if mask.sum() > 0:
                y_true = df.loc[mask, target].values
                y_pred = df.loc[mask, f"Predicted_{target}"].values
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"{target} (n={mask.sum()}): MAE = {mae:.2f} K, R² = {r2:.3f}")
            else:
                print(f"{target}: No actual values available for comparison")
    else:
        print("No samples with all target values available for visualization")
    
    return df

def predict_new_data(input_data_path, output_path=None):
    """
    Make predictions on new data using the trained model
    
    Parameters:
    -----------
    input_data_path : str
        Path to the CSV file with featurized data
    output_path : str, optional
        Path to save the predictions. If None, saves to default location.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with original data plus predictions
    """
    print("\n=== Making predictions on new data ===")
    
    # Check if model exists
    if not os.path.exists(STAGE1_MODEL):
        raise FileNotFoundError(f"Model not found at {STAGE1_MODEL}. Train the model first.")
    
    # Load the new data
    df = pd.read_csv(input_data_path)
    print(f"Loaded {len(df)} samples from {input_data_path}")
    
    # Load the trained model components
    top_features = np.load(STAGE1_TOP)
    scaler = joblib.load(STAGE1_SCALER)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in TARGET_NAMES and np.issubdtype(df[c].dtype, np.number)]
    
    # Make sure all top features exist in the dataset
    missing_features = [f for f in top_features if f not in feature_cols]
    if missing_features:
        print(f"Warning: {len(missing_features)} features are missing from the dataset")
        # Only use features that exist in both
        common_features = [f for f in top_features if f in feature_cols]
        print(f"Using {len(common_features)} common features")
        top_features = np.array(common_features)
    
    # Prepare features
    X = df[top_features].fillna(0).values
    X_scaled = scaler.transform(X)
    
    # Load model
    model = GlassThermalModel(input_dim=len(top_features))
    model.load_state_dict(torch.load(STAGE1_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Make predictions in batches
    predictions = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(DEVICE)
            batch_pred = model(batch).cpu().numpy()
            predictions.append(batch_pred)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Add predictions to dataframe
    for i, target in enumerate(TARGET_NAMES):
        df[f'Predicted_{target}'] = predictions[:, i]
    
    # Save predictions
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "new_data_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Calculate metrics only for samples with non-NaN actual values
    print("\n=== Prediction Summary ===")
    print(f"Total samples predicted: {len(df)}")
    print("\nPrediction vs Actual comparison (only for samples with actual values):")
    
    for target in TARGET_NAMES:
        mask = df[target].notna()
        if mask.sum() > 0:
            y_true = df.loc[mask, target].values
            y_pred = df.loc[mask, f"Predicted_{target}"].values
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"{target} (n={mask.sum()}): MAE = {mae:.2f} K, R² = {r2:.3f}")
        else:
            print(f"{target}: No actual values available for comparison")
    
    return df

# ---------------- Main Training ----------------
def main_train():
    print("Loading featurized data...")
    df = pd.read_csv(FEATURIZED_CSV)

    # Filter for physically valid samples
    print("Filtering training data: requiring Tg + 25 < Tx and Tx + 25 < Tl...")
    valid_mask = (
        (df['Tg'] < df['Tx'] - MIN_MARGIN) &
        (df['Tx'] < df['Tl'] - MIN_MARGIN) &
        df[TARGET_NAMES].notna().all(axis=1)
    )
    df_train = df[valid_mask].reset_index(drop=True)
    print(f"Training samples: {len(df_train)} / {len(df)}")
    
    # Save training data info
    training_info = {
        'training_samples': len(df_train),
        'total_samples': len(df)
    }

    feature_cols = [c for c in df_train.columns if c not in TARGET_NAMES and np.issubdtype(df_train[c].dtype, np.number)]
    X_full = df_train[feature_cols].fillna(0).values
    y_full = df_train[TARGET_NAMES].values

    print("\nSelecting top features...")
    selector = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    selector.fit(X_full, y_full)
    top_idx = np.argsort(selector.feature_importances_)[-TOP_FEATURES:]
    top_feature_names = np.array(feature_cols)[top_idx]
    np.save(STAGE1_TOP, top_feature_names)
    print(f"Saved top features → {STAGE1_TOP}")
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

    X_selected = X_full[:, top_idx]

    scaler = StandardScaler()
    scaler.fit(X_selected)
    joblib.dump(scaler, STAGE1_SCALER)
    print(f"Saved scaler → {STAGE1_SCALER}")
    
    # Fit PCA on the same data used for the scaler and save it
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    pca.fit(X_selected)
    joblib.dump(pca, os.path.join(OUTPUT_DIR, "pca_object.joblib"))
    print(f"Saved PCA object → {os.path.join(OUTPUT_DIR, 'pca_object.joblib')}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_global_mae = float('inf')
    best_fold_preds, best_fold_trues = None, None
    best_model = None

    print("\nStarting cross-validation training...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_selected), 1):
        X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
        y_tr, y_val = y_full[tr_idx], y_full[val_idx]

        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_tr)
        X_val_s = scaler_fold.transform(X_val)

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr)),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val)),
                                batch_size=BATCH_SIZE)

        model = GlassThermalModel(input_dim=TOP_FEATURES)
        model, fold_mae, f_preds, f_trues = train_one_fold(model, train_loader, val_loader, fold)

        if fold_mae < best_global_mae:
            best_global_mae = fold_mae
            best_fold_preds, best_fold_trues = f_preds, f_trues
            best_model = model
            torch.save(model.state_dict(), STAGE1_MODEL)
            print(f"*** NEW BEST MODEL SAVED (Fold {fold}, MAE: {fold_mae:.2f} K) ***")

    print(f"\nTraining complete. Best model global MAE: {best_global_mae:.2f} K")
    
    # Generate visualization plots using data from the best fold
    if best_fold_preds is not None:
        print("\nGenerating training visualization plots...")
        plot_parity_publication(best_fold_trues, best_fold_preds)
        plot_residuals_summary(best_fold_trues, best_fold_preds)
    
    # Save all training results and model elements
    if best_model is not None:
        save_training_results(best_model, best_fold_preds, best_fold_trues, 
                            scaler, top_feature_names, best_global_mae, training_info)

    print(f"\nBest model artifact: {STAGE1_MODEL}")
    print(f"All model elements saved in folder: {OUTPUT_DIR}")
    
    # Make predictions on the entire dataset
    predict_on_full_dataset()
    
    print("\n=== All tasks completed successfully! ===")

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined thermal regression model for metallic glasses")
    parser.add_argument("--mode", choices=["train", "predict", "predict-full"], default="train",
                       help="Mode: 'train' to train model, 'predict' for new data, 'predict-full' for full dataset")
    parser.add_argument("--input", type=str, help="Input CSV file for prediction mode")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main_train()
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nTo make predictions on new data, use:")
        print("  python thermal_regression_streamlined.py --mode predict --input new_data.csv --output predictions.csv")
        print("\nTo make predictions on the full dataset, use:")
        print("  python thermal_regression_streamlined.py --mode predict-full")
        print("\nOr in Python code:")
        print("  from thermal_regression_streamlined import predict_new_data")
        print("  predictions = predict_new_data('new_data.csv', 'predictions.csv')")
        
    elif args.mode == "predict":
        if not args.input:
            print("Error: --input argument required for prediction mode")
            print("Usage: python thermal_regression_streamlined.py --mode predict --input data.csv --output predictions.csv")
        else:
            predictions = predict_new_data(args.input, args.output)
            print("\n" + "="*60)
            print("PREDICTION COMPLETE!")
            print("="*60)
            
    elif args.mode == "predict-full":
        predict_on_full_dataset()
        print("\n" + "="*60)
        print("FULL DATASET PREDICTION COMPLETE!")
        print("="*60)
