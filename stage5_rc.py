import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import joblib
import warnings
import argparse
import json

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAME = "Rc"
FEAT_CSV = "featurized_metallic_glass_stage1.csv"
TARGETS_CSV = "targets_full.csv"
OUT_DIR = f"improved_{TARGET_NAME}_v1"
os.makedirs(OUT_DIR, exist_ok=True)

# Optimized for small N=71
K_FOLDS = 5
N_REPEATS = 5  # Repeated CV for stability
EPOCHS = 1000
BATCH_SIZE = 8 # Smaller batch for small N
MAX_LR = 5e-4
WEIGHT_DECAY = 0.05 # Higher decay to prevent overfitting
PATIENCE = 100
TOP_K_FEATURES = 12 # Reduced feature count for generalization

DROP_COLS = ['Phase', 'Tg', 'Tx', 'Tl', 'Rc', 'Dmax', 'gamma', 'Trg', 'delta_Tx', 
             'Alloys', 'composition_obj', 'Condition_Status', 'Tg_new', 'Tx_new', 'Tl_new']

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------- Model Architecture ----------------
class RobustMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simpler architecture is better for N=71
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.LayerNorm(24),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(24, 12),
            nn.SiLU(),
            nn.Linear(12, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Visualization ----------------
def plot_rc_parity(y_true, y_pred, save_path=None):
    plt.figure(figsize=(7, 6))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Experimental Log10(Rc)")
    plt.ylabel("Predicted Log10(Rc)")
    plt.title(f"Ensemble Prediction (N=71)\n$R^2$={r2:.3f}, MAE={mae:.3f}")
    
    if save_path:
        plt.savefig(save_path)
        print(f"\n[INFO] Parity plot saved as '{save_path}'")
    else:
        plt.savefig(os.path.join(OUT_DIR, "improved_rc_parity.png"))
        print(f"\n[INFO] Parity plot saved as 'improved_rc_parity.png'")
    
    plt.show()

# ---------------- Preprocessing for Prediction ----------------
def preprocess_data_for_prediction(X_data, artifacts):
    """
    Apply the same preprocessing as training
    """
    # Unpack artifacts
    selected_cols = artifacts["selected_cols"]
    scaler = artifacts["scaler"]
    
    # Select the same features used in training
    X_selected = X_data[selected_cols]
    
    # Apply median imputation (same as in training)
    X_selected = X_selected.fillna(X_selected.median())
    
    # Apply scaling
    X_scaled = scaler.transform(X_selected)
    
    return X_scaled

# ---------------- Main Training Function ----------------
def train_model():
    """
    Train the model and save all artifacts
    """
    print("=== TRAINING MODE ===")
    print(f"--- Improved {TARGET_NAME} Training (N=71) ---")
    
    df_feat = pd.read_csv(FEAT_CSV)
    df_targets = pd.read_csv(TARGETS_CSV)
    
    # Save original data for later
    original_df = df_feat.copy()

    # 1. Data Cleaning
    y_raw = pd.to_numeric(df_targets[TARGET_NAME], errors='coerce').values
    valid_mask = (y_raw > 0) & np.isfinite(y_raw)
    
    X_raw = df_feat[valid_mask].drop(columns=[c for c in DROP_COLS if c in df_feat.columns], errors='ignore')
    X_raw = X_raw.select_dtypes(include=[np.number])
    y = np.log10(y_raw[valid_mask])
    
    # Save the valid indices for later
    valid_indices = df_feat.index[valid_mask].tolist()
    
    # Also save the original Rc values for these indices
    original_rc_values = y_raw[valid_mask]

    # 2. Median Imputation (Crucial for small physical datasets)
    X_raw = X_raw.fillna(X_raw.median())
    
    # 3. Feature Selection
    print(f"Selecting top {TOP_K_FEATURES} stable features...")
    selector = ExtraTreesRegressor(n_estimators=500, random_state=42)
    selector.fit(X_raw, y)
    top_idx = np.argsort(selector.feature_importances_)[-TOP_K_FEATURES:]
    selected_cols = X_raw.columns[top_idx].tolist()
    X_selected = X_raw[selected_cols].values
    
    # Save feature importances
    feat_importance = pd.DataFrame({'Feature': selected_cols, 'Importance': selector.feature_importances_[top_idx]})
    feat_importance.sort_values('Importance', ascending=False).to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    
    # 4. Repeated Cross-Validation
    rkf = RepeatedKFold(n_splits=K_FOLDS, n_repeats=N_REPEATS, random_state=42)
    all_preds, all_trues = [], []
    
    # Track the best models
    best_mlp_model = None
    best_gbr_model = None
    best_scaler = None
    best_global_mae = float('inf')

    for run, (tr_idx, val_idx) in enumerate(rkf.split(X_selected), 1):
        X_tr, X_val = X_selected[tr_idx], X_selected[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # A: Train MLP
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).view(-1,1)), 
                                  batch_size=BATCH_SIZE, shuffle=True)
        model = RobustMLP(input_dim=TOP_K_FEATURES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.HuberLoss(delta=1.0) # Robust to outliers

        best_val = float('inf')
        patience_cnt = 0
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                v_pred = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy().flatten()
                v_mae = mean_absolute_error(y_val, v_pred)
                if v_mae < best_val:
                    best_val = v_mae
                    patience_cnt = 0
                    best_mlp_state = model.state_dict()
                else: patience_cnt += 1
            if patience_cnt >= PATIENCE: break

        model.load_state_dict(best_mlp_state)
        mlp_preds = model(torch.FloatTensor(X_val).to(DEVICE)).cpu().detach().numpy().flatten()

        # B: Train Tree Baseline (HistGradientBoosting handles noise well)
        gbr = HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=42)
        gbr.fit(X_tr, y_tr)
        gbr_preds = gbr.predict(X_val)

        # C: Ensemble (Weighted average)
        # Tree-based models are usually more accurate on this small N
        combined_preds = (0.3 * mlp_preds) + (0.7 * gbr_preds)
        
        # Calculate ensemble MAE
        ensemble_mae = mean_absolute_error(y_val, combined_preds)
        
        # Track the best models
        if ensemble_mae < best_global_mae:
            best_global_mae = ensemble_mae
            best_mlp_model = RobustMLP(input_dim=TOP_K_FEATURES)
            best_mlp_model.load_state_dict(best_mlp_state)
            best_gbr_model = gbr
            best_scaler = scaler
        
        all_preds.append(combined_preds)
        all_trues.append(y_val)
        if run % 5 == 0: print(f"Completed CV Fold-Run {run}/{K_FOLDS*N_REPEATS}")

    # --- Evaluation ---
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    
    r2 = r2_score(all_trues, all_preds)
    mae = mean_absolute_error(all_trues, all_preds)
    print(f"\n[FINAL ENSEMBLE RESULTS] R2: {r2:.4f} | MAE: {mae:.4f}")

    # Plotting
    plot_rc_parity(all_trues, all_preds)
    
    # Save all artifacts
    print("\nSaving all artifacts for prediction...")
    artifacts = {
        "mlp_model": best_mlp_model,
        "gbr_model": best_gbr_model,
        "scaler": best_scaler,
        "selected_cols": selected_cols,
        "random_seed": 42
    }
    
    # Save models
    torch.save(best_mlp_model.state_dict(), os.path.join(OUT_DIR, "mlp_model.pth"))
    joblib.dump(best_gbr_model, os.path.join(OUT_DIR, "gbr_model.joblib"))
    
    # Save other artifacts
    joblib.dump({
        "selected_cols": selected_cols,
        "scaler": best_scaler,
        "input_dim": TOP_K_FEATURES
    }, os.path.join(OUT_DIR, "artifacts.joblib"))
    
    # Save training config
    config = {
        "random_seed": 42,
        "k_folds": K_FOLDS,
        "n_repeats": N_REPEATS,
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
    
    with open(os.path.join(OUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4, default=convert_numpy)
    
    # Predict on entire dataset
    print("\n=== Making predictions on the entire dataset ===")
    # Filter to only include rows with valid indices
    df_valid = original_df.iloc[valid_indices].copy()
    
    # Prepare features
    feature_cols = [c for c in df_valid.columns if c not in DROP_COLS and np.issubdtype(df_valid[c].dtype, np.number)]
    X_for_prediction = df_valid[feature_cols].fillna(df_valid[feature_cols].median())
    
    # Use the trained model to predict on the entire dataset
    X_processed = preprocess_data_for_prediction(X_for_prediction, artifacts)
    
    # Load models and make predictions
    mlp_model = best_mlp_model.to(DEVICE)
    mlp_model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
        mlp_preds = mlp_model(X_tensor).cpu().numpy().flatten()
    
    gbr_preds = best_gbr_model.predict(X_processed)
    
    # Ensemble predictions
    predictions_log = (0.3 * mlp_preds) + (0.7 * gbr_preds)
    
    # Convert back from log scale
    predictions = 10 ** predictions_log
    
    # Add predictions to dataframe
    df_valid["Predicted_Rc"] = predictions
    df_valid["Predicted_Rc_log"] = predictions_log
    
    # Add the original Rc values directly to avoid merge issues
    df_valid["Original_Rc"] = original_rc_values
    
    # Save predictions to CSV
    predictions_path = os.path.join(OUT_DIR, "predictions_on_full_dataset.csv")
    df_valid.to_csv(predictions_path, index=False)
    
    # Create a summary of predictions
    print("\n=== Prediction Summary ===")
    print(f"Total samples predicted: {len(df_valid)}")
    
    # Compare with original values - FIX: Use the values we already have
    if 'Original_Rc' in df_valid.columns:
        # Filter out any non-positive values for comparison
        valid_comparison = df_valid['Original_Rc'] > 0
        
        if valid_comparison.any():
            mae = mean_absolute_error(np.log10(df_valid.loc[valid_comparison, 'Original_Rc']), 
                                     df_valid.loc[valid_comparison, 'Predicted_Rc_log'])
            r2 = r2_score(np.log10(df_valid.loc[valid_comparison, 'Original_Rc']), 
                         df_valid.loc[valid_comparison, 'Predicted_Rc_log'])
            print(f"Rc: MAE (log) = {mae:.3f}, R² (log) = {r2:.3f}")
    
    print(f"\nPredictions saved to: {predictions_path}")
    print("Done - Final model saved:", OUT_DIR)
    
    return df_valid

# ---------------- Standalone Prediction Function ----------------
def predict_new_data(input_data, output_file=None, artifacts_dir=f"improved_{TARGET_NAME}_v1"):
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
    feature_cols = [c for c in df.columns if c not in DROP_COLS and np.issubdtype(df[c].dtype, np.number)]
    X_for_prediction = df[feature_cols].fillna(df[feature_cols].median())
    
    # Preprocess
    X_processed = preprocess_data_for_prediction(X_for_prediction, artifacts)
    
    # Load models and make predictions
    mlp_model = RobustMLP(input_dim=X_processed.shape[1])
    mlp_model.load_state_dict(torch.load(os.path.join(artifacts_dir, "mlp_model.pth")))
    mlp_model.to(DEVICE)
    mlp_model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(DEVICE)
        mlp_preds = mlp_model(X_tensor).cpu().numpy().flatten()
    
    gbr_model = joblib.load(os.path.join(artifacts_dir, "gbr_model.joblib"))
    gbr_preds = gbr_model.predict(X_processed)
    
    # Ensemble predictions
    predictions_log = (0.3 * mlp_preds) + (0.7 * gbr_preds)
    
    # Convert back from log scale
    predictions = 10 ** predictions_log
    
    # Add predictions to dataframe
    original_df["Predicted_Rc"] = predictions
    original_df["Predicted_Rc_log"] = predictions_log
    
    # Create summary
    print(f"\nPrediction Summary:")
    print(f"Total samples predicted: {len(original_df)}")
    print(f"Rc range: {predictions.min():.2f} to {predictions.max():.2f}")
    
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
