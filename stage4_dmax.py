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
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5
EPOCHS = 1000
BATCH_SIZE = 32
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 100
RANDOM_SEED = 42
TOP_K_FEATURES = 50

# Create output directory
OUTPUT_DIR = "STAGE4_DMAX_REGRESSOR"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
FEAT_CSV = "featurized_metallic_glass_stage1.csv"
TARGETS_CSV = "targets_full.csv"
DMAX_MODEL = os.path.join(OUTPUT_DIR, "dmax_model.pth")
DMAX_FEATURES = os.path.join(OUTPUT_DIR, "dmax_features.npy")
DMAX_SCALER = os.path.join(OUTPUT_DIR, "dmax_scaler.joblib")
PARITY_PLOT = os.path.join(OUTPUT_DIR, "dmax_parity_log.png")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ---------------- Model Architecture ----------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return F.silu(x + self.block(x))

class DmaxResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, 0.3),
            ResidualBlock(hidden_dim, 0.2),
            ResidualBlock(hidden_dim, 0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.head(x)

# ---------------- Visualization ----------------
def plot_dmax_parity(y_true_log, y_pred_log, save_path=None):
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    r2 = r2_score(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae = mean_absolute_error(y_true_log, y_pred_log)

    sns.regplot(x=y_true_log, y=y_pred_log, ax=ax, 
                scatter_kws={'alpha':0.5, 's':50, 'color':'#4575b4'},
                line_kws={'color': 'black', 'linestyle': '--'})
    
    lims = [y_true_log.min() - 0.2, y_true_log.max() + 0.2]
    ax.plot(lims, lims, color='gray', lw=1.5, zorder=1)

    ax.set_title("Critical Diameter ($D_{max}$) Prediction", fontsize=20, pad=15)
    ax.set_xlabel("Experimental $\log_{10}(D_{max})$ [mm]", fontsize=16)
    ax.set_ylabel("Predicted $\log_{10}(D_{max})$ [mm]", fontsize=16)
    
    stats_text = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$\n$MAE = {mae:.3f}$"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"\n[INFO] Parity plot saved as '{save_path}'")
    else:
        plt.savefig(PARITY_PLOT)
        print(f"\n[INFO] Parity plot saved as '{PARITY_PLOT}'")
    
    plt.show()

# ---------------- Preprocessing for Prediction ----------------
def preprocess_data_for_prediction(X_data, artifacts):
    """
    Apply the same preprocessing as training
    """
    # Unpack artifacts
    top_features = artifacts["top_features"]
    scaler = artifacts["scaler"]
    
    # Select the same features used in training
    X_selected = X_data[top_features]
    
    # Apply scaling
    X_scaled = scaler.transform(X_selected)
    
    # Clean the data (same as in training)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_scaled

# ---------------- Main Training Function ----------------
def train_model():
    """
    Train the model and save all artifacts
    """
    print("=== TRAINING MODE ===")
    print("Loading data...")
    if not os.path.exists(FEAT_CSV) or not os.path.exists(TARGETS_CSV):
        print(f"[ERROR] Required CSV files not found.")
        return None

    df_feat = pd.read_csv(FEAT_CSV)
    df_targets = pd.read_csv(TARGETS_CSV)
    
    # Save original data for later
    original_df = df_feat.copy()

    target_col = 'Dmax'
    df_targets[target_col] = pd.to_numeric(df_targets[target_col], errors='coerce')
    
    # Ensure Dmax > 0 to avoid -inf in log10
    valid_mask = (df_targets[target_col] > 0) & df_targets[target_col].notna()
    X = df_feat[valid_mask].select_dtypes(include=[np.number]).drop(columns=['Tg','Tx','Tl'], errors='ignore')
    y = np.log10(df_targets.loc[valid_mask, target_col].values)
    
    # Save the valid indices for later
    valid_indices = df_feat.index[valid_mask].tolist()
    
    # Also save the original Dmax values for these indices
    original_dmax_values = df_targets.loc[valid_indices, target_col].values

    print(f"Selecting top {TOP_K_FEATURES} features...")
    selector = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    # Fill internal NaNs for feature selection
    selector.fit(X.fillna(0), y)
    top_idx = np.argsort(selector.feature_importances_)[-TOP_K_FEATURES:]
    top_feature_names = X.columns[top_idx].tolist()
    X_selected = X.iloc[:, top_idx].values
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
    
    # CRITICAL: Clean selected features for the Neural Network
    X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_preds, all_trues = [], []
    best_global_mae = float('inf')
    best_model = None

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        print(f"\n--- Starting Fold {fold} ---")
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).view(-1,1)), 
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1,1)), 
                                batch_size=BATCH_SIZE)

        model = DmaxResidualNet(input_dim=TOP_K_FEATURES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
        criterion = nn.MSELoss()

        best_val_mae = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                # Gradient clipping to prevent 'inf' weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_mae_accum = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb.to(DEVICE))
                    val_mae_accum += F.l1_loss(pred, yb.to(DEVICE)).item() * xb.size(0)
            
            avg_val_mae = val_mae_accum / len(val_idx)
            scheduler.step(avg_val_mae)

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if patience_cnt >= PATIENCE:
                break

        print(f"Fold {fold} | Best Val MAE (log): {best_val_mae:.4f}")
        
        # Load the best state for this fold to generate predictions
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_fold_pred = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
            all_preds.append(final_fold_pred.flatten())
            all_trues.append(y_val)
            
        # Track the best model across folds
        if best_val_mae < best_global_mae:
            best_global_mae = best_val_mae
            best_model = DmaxResidualNet(input_dim=TOP_K_FEATURES)
            best_model.load_state_dict(best_state)

    # Final Evaluation across all folds
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    plot_dmax_parity(all_trues, all_preds)
    
    # Save all artifacts
    print("\nSaving all artifacts for prediction...")
    artifacts = {
        "model_state_dict": best_model.state_dict(),
        "top_features": top_feature_names,
        "scaler": scaler,
        "random_seed": RANDOM_SEED
    }
    
    # Save model
    torch.save(best_model.state_dict(), DMAX_MODEL)
    
    # Save other artifacts
    joblib.dump({
        "top_features": top_feature_names,
        "scaler": scaler,
        "input_dim": TOP_K_FEATURES
    }, os.path.join(OUTPUT_DIR, "artifacts.joblib"))
    
    # Save training config
    config = {
        "random_seed": RANDOM_SEED,
        "k_folds": K_FOLDS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_lr": MAX_LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "top_features": TOP_K_FEATURES,
        "best_global_mae": best_global_mae
    }
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4, default=convert_numpy)
    
    # Predict on entire dataset
    print("\n=== Making predictions on the entire dataset ===")
    # Filter to only include rows with valid indices
    df_valid = original_df.iloc[valid_indices].copy()
    
    # Prepare features
    feature_cols = [c for c in df_valid.columns if c not in ['Tg','Tx','Tl'] and np.issubdtype(df_valid[c].dtype, np.number)]
    X_for_prediction = df_valid[feature_cols].fillna(0)
    
    # Use the trained model to predict on the entire dataset
    X_processed = preprocess_data_for_prediction(X_for_prediction, artifacts)
    
    # Load model and make predictions
    model = DmaxResidualNet(input_dim=X_processed.shape[1])
    model.load_state_dict(best_model.state_dict())
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
        predictions_log = model(X_tensor).cpu().numpy().flatten()
    
    # Convert back from log scale
    predictions = 10 ** predictions_log
    
    # Add predictions to dataframe
    df_valid["Predicted_Dmax"] = predictions
    df_valid["Predicted_Dmax_log"] = predictions_log
    
    # Add the original Dmax values directly to avoid merge issues
    df_valid["Original_Dmax"] = original_dmax_values
    
    # Save predictions to CSV
    predictions_path = os.path.join(OUTPUT_DIR, "predictions_on_full_dataset.csv")
    df_valid.to_csv(predictions_path, index=False)
    
    # Create a summary of predictions
    print("\n=== Prediction Summary ===")
    print(f"Total samples predicted: {len(df_valid)}")
    
    # Compare with original values - FIX: Use the values we already have
    if 'Original_Dmax' in df_valid.columns:
        # Filter out any non-positive values for comparison
        valid_comparison = df_valid['Original_Dmax'] > 0
        
        if valid_comparison.any():
            mae = mean_absolute_error(np.log10(df_valid.loc[valid_comparison, 'Original_Dmax']), 
                                     df_valid.loc[valid_comparison, 'Predicted_Dmax_log'])
            r2 = r2_score(np.log10(df_valid.loc[valid_comparison, 'Original_Dmax']), 
                         df_valid.loc[valid_comparison, 'Predicted_Dmax_log'])
            print(f"Dmax: MAE (log) = {mae:.3f}, R² (log) = {r2:.3f}")
    
    print(f"\nPredictions saved to: {predictions_path}")
    print("Done - Final model saved:", OUTPUT_DIR)
    
    return df_valid

# ---------------- Standalone Prediction Function ----------------
def predict_new_data(input_data, output_file=None, artifacts_dir="STAGE4_DMAX_REGRESSOR"):
    """
    Standalone function to predict on new data
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        - Path to CSV file with featurized data
        - Or DataFrame with featurized data
    output_file : str, optional
        Path to save predictions CSV
    artifacts_dir : str
        Directory containing trained models
    
    Returns:
    --------
    DataFrame with predictions added
    """
    print(f"\n=== PREDICTION MODE ===")
    
    # Load data
    if isinstance(input_data, str):
        print(f"Loading data from: {input_data}")
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    
    # Save original
    original_df = df.copy()
    
    # Load artifacts
    artifacts_path = os.path.join(artifacts_dir, "artifacts.joblib")
    artifacts = joblib.load(artifacts_path)
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['Tg','Tx','Tl'] and np.issubdtype(df[c].dtype, np.number)]
    X_for_prediction = df[feature_cols].fillna(0)
    
    # Preprocess
    X_processed = preprocess_data_for_prediction(X_for_prediction, artifacts)
    
    # Load model and make predictions
    model = DmaxResidualNet(input_dim=X_processed.shape[1])
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, "dmax_model.pth")))
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
        predictions_log = model(X_tensor).cpu().numpy().flatten()
    
    # Convert back from log scale
    predictions = 10 ** predictions_log
    
    # Add predictions to dataframe
    original_df["Predicted_Dmax"] = predictions
    original_df["Predicted_Dmax_log"] = predictions_log
    
    # Create summary
    print(f"\nPrediction Summary:")
    print(f"Total samples predicted: {len(original_df)}")
    print(f"Dmax range: {predictions.min():.2f} to {predictions.max():.2f} mm")
    
    # Save to file
    if output_file:
        original_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    
    return original_df

# -------------------- USAGE --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model or make predictions")
    parser.add_argument("--mode", choices=["train", "predict"], default="train",
                       help="Mode: 'train' to train model, 'predict' to make predictions")
    parser.add_argument("--input", type=str, help="Input CSV file for prediction")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train the model
        results = train_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nTo make predictions on new data, use:")
        print("  python script.py --mode predict --input new_data.csv --output predictions.csv")
        print("\nOr in Python code:")
        print("  from script_name import predict_new_data")
        print("  predictions = predict_new_data('new_data.csv', 'output.csv')")
        
    elif args.mode == "predict":
        if not args.input:
            print("Error: --input argument required for prediction mode")
            print("Usage: python script.py --mode predict --input data.csv --output predictions.csv")
        else:
            # Make predictions
            predictions = predict_new_data(args.input, args.output)
            
            print("\n" + "="*60)
            print("PREDICTION COMPLETE!")
            print("="*60)
