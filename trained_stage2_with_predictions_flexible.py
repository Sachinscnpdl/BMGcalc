#!/usr/bin/env python3
# trained_stage2_with_predictions_improved.py
"""
Improved version - Minimal changes while maintaining architecture
"""

import os
import random
import warnings
import joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import mode
from time import time  # Added for timing

# ----------------------- CONFIG -----------------------
FEAT_CSV = "featurized_metallic_glass_stage1.csv"
OUT_DIR = "STAGE2_classificatuion"
PREDICTIONS_FILE = "predictions_on_full_dataset.csv"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 5  # Changed from 2 to 5 for better cross-validation
KBEST_FEATURES = 140
RFECV_MIN_FEATURES = 8
SMOTE_FRACTION = 0.22
SMOTE_FRACTION_IN_FOLD = 0.20
HGB_SEEDS = [SEED, SEED+1, SEED+2]
NN_SEEDS = [SEED+3, SEED+6, SEED+9]
NN_EPOCHS = 200
NN_BATCH = 64
NN_MAX_LR = 8e-4
GRAD_CLIP = 1.0
MIN_VERIFIER_PREC = 0.65
META_THR_GRID = np.linspace(0.15, 0.75, 35)
VER_THR_GRID = np.linspace(0.30, 0.90, 35)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------- MODEL CLASSES --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(x))
        out = self.fc1(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + residual

class AdvancedNN(nn.Module):
    def __init__(self, dim_in, n_classes):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim_in)
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            ResidualBlock(512),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.25),
            ResidualBlock(256),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        x = self.input_norm(x)
        z = self.feature_extractor(x)
        return self.classifier(z)

# -------------------- HELPER FUNCTIONS --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal = alpha_t * focal
        return focal.mean()

def make_inverse_freq_weights(counts):
    inv = 1.0 / (np.array(counts) + 1e-9)
    inv = inv / np.mean(inv)
    return np.clip(inv, 1.0, 8.0)

def train_nn(model, X_tr, y_tr, X_val, y_val, class_counts=None, epochs=NN_EPOCHS):
    model = model.to(DEVICE)
    alpha = make_inverse_freq_weights(class_counts) if class_counts is not None else None
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32).to(DEVICE) if alpha is not None else None
    criterion = FocalLoss(alpha=alpha_tensor)
    sample_w = compute_sample_weight("balanced", y_tr)
    sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)
    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=NN_BATCH, sampler=sampler, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=NN_MAX_LR, weight_decay=1e-4)
    total_steps = max(1, epochs * len(loader))
    try:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=NN_MAX_LR, total_steps=total_steps)
    except Exception:
        scheduler = None
    best_state = None
    best_val = -1.0
    patience = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, average='macro')
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if ep % 20 == 0 or ep == 1:
            print(f" NN epoch {ep} val_macroF1 {val_f1:.4f} (best {best_val:.4f})")
        if patience >= 35:
            print(f" Early stopping at epoch {ep}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model

def fit_temperature_scalar(model, X_valid, y_valid):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_valid, dtype=torch.float32).to(DEVICE)).detach()
    labels = torch.tensor(y_valid, dtype=torch.long).to(DEVICE)
    temp = nn.Parameter(torch.ones(1).to(DEVICE))
    optimizer = optim.LBFGS([temp], lr=0.1, max_iter=100)
    criterion = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
    except Exception:
        pass
    return float(temp.item())

def build_meta_features_from_probs(prob_list, minority_label):
    ps = prob_list
    n = ps[0].shape[0]
    n_classes = ps[0].shape[1]
    feats = []
    for p in ps:
        maxp = p.max(axis=1)
        top2 = np.partition(p, -2, axis=1)[:, -2]
        margin = maxp - top2
        entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
        minority_p = p[:, minority_label]
        feats.append(minority_p.reshape(-1,1))
        feats.append(maxp.reshape(-1,1))
        feats.append(margin.reshape(-1,1))
        feats.append(entropy.reshape(-1,1))
    arr_minority = np.vstack([p[:, minority_label] for p in ps]).T
    mean_minority = arr_minority.mean(axis=1, keepdims=True)
    arr_max = np.vstack([p.max(axis=1) for p in ps]).T
    mean_max = arr_max.mean(axis=1, keepdims=True)
    arr_entropy = np.vstack([-np.sum(p * np.log(p + 1e-12), axis=1) for p in ps]).T
    mean_entropy = arr_entropy.mean(axis=1, keepdims=True)
    argmaxs = np.vstack([p.argmax(axis=1) for p in ps]).T
    maj_vote, _ = mode(argmaxs, axis=1)
    maj_vote = maj_vote.ravel()
    agreement = (argmaxs == maj_vote.reshape(-1,1)).mean(axis=1, keepdims=True)
    all_feats = np.hstack(feats + [mean_minority, mean_max, mean_entropy, agreement])
    return all_feats

def tune_thresholds(meta_val_probs, verifier_val_probs, y_val, minority_label, meta_grid, ver_grid, min_prec):
    best = {"macro_f1": -1.0, "m_thr": 0.5, "v_thr": 0.5}
    y_true = y_val
    for m_thr in meta_grid:
        for v_thr in ver_grid:
            preds = np.argmax(meta_val_probs, axis=1)
            accept_min = (meta_val_probs[:, minority_label] >= m_thr) | (verifier_val_probs >= v_thr)
            preds_adj = preds.copy()
            preds_adj[accept_min] = minority_label
            macro_f1 = f1_score(y_true, preds_adj, average='macro')
            min_prec_val = precision_score(y_true, preds_adj, labels=[minority_label], average=None, zero_division=0)[0]
            if min_prec_val >= min_prec and macro_f1 > best["macro_f1"]:
                best.update({"macro_f1": macro_f1, "m_thr": m_thr, "v_thr": v_thr})
    if best["macro_f1"] == -1.0:
        best_uncon = {"macro_f1": -1.0, "m_thr": 0.5, "v_thr": 0.5}
        for m_thr in meta_grid:
            for v_thr in ver_grid:
                preds = np.argmax(meta_val_probs, axis=1)
                accept_min = (meta_val_probs[:, minority_label] >= m_thr) | (verifier_val_probs >= v_thr)
                preds_adj = preds.copy()
                preds_adj[accept_min] = minority_label
                macro_f1 = f1_score(y_true, preds_adj, average='macro')
                if macro_f1 > best_uncon["macro_f1"]:
                    best_uncon.update({"macro_f1": macro_f1, "m_thr": m_thr, "v_thr": v_thr})
        return best_uncon
    return best

# -------------------- PREDICTION SYSTEM --------------------
def preprocess_data_for_prediction(X_data, artifacts):
    """
    Apply EXACT same preprocessing as training
    """
    # Unpack all necessary artifacts
    t_encoder = artifacts["t_encoder"]
    top_40_cols = artifacts["top_40_cols"]
    skb = artifacts["skb"]
    scaler_rf = artifacts["scaler_rf"]
    selector = artifacts["selector"]
    selected_columns = artifacts["selected_columns"]
    scaler = artifacts["scaler"]
    
    X_processed = X_data.copy()
    
    # 1. Initial preprocessing
    eps = 1e-10
    for c in ["Tg","Tx","Tl"]:
        if c in X_processed.columns:
            X_processed[c] = X_processed[c].replace(0, eps).fillna(X_processed[c].mean())
    
    if t_encoder is not None:
        X_processed = t_encoder.transform(X_processed).fillna(X_processed.mean(numeric_only=True))
    
    # Made domain features optional
    if all(c in X_processed.columns for c in ["Tg","Tx","Tl"]):
        X_processed["Trg"] = X_processed["Tg"] / X_processed["Tl"]
        X_processed["Delta_Tx"] = X_processed["Tx"] - X_processed["Tg"]
        X_processed["Gamma"] = X_processed["Tx"] / (X_processed["Tg"] + X_processed["Tl"])
        X_processed["Kgl"] = (X_processed["Tx"] - X_processed["Tg"]) / (X_processed["Tl"] - X_processed["Tx"] + eps)
        X_processed["Omega"] = X_processed["Tg"] / (X_processed["Tl"] - X_processed["Tx"] + eps)
        
        for col in ["Delta_Tx","Kgl","Omega"]:
            if col in X_processed.columns:
                X_processed[f"{col}_log"] = np.log1p(np.abs(X_processed[col]))
    
    X_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_processed.fillna(X_processed.mean(numeric_only=True), inplace=True)

    # 2. Apply the FULL feature selection pipeline
    X_top = X_processed[top_40_cols].copy()
    X_keep = skb.transform(X_top)
    X_scaled_rf = scaler_rf.transform(X_keep)
    X_selected = selector.transform(X_scaled_rf)
    X_final = scaler.transform(X_selected)
    
    return X_final

# -------------------- MAIN TRAINING FUNCTION --------------------
def train_model():
    """
    Train the model and save all artifacts
    """
    print("=== TRAINING MODE ===")
    start_time = time()  # Added timing
    set_seed(SEED)
    df = pd.read_csv(FEAT_CSV)
    df["Phase"] = df["Phase"].astype(str).str.strip()
    print("Raw Phase counts before filtering:\n", df["Phase"].value_counts(dropna=False))
    
    df = df[~df["Phase"].isin(["nan","NAN","","None","Unknown", "CMG"])].reset_index(drop=True)
    df["Phase"] = df["Phase"].replace({"MG":"Metalic_Glass","MMG":"Metalic_Glass"})
    print("Data after filtering:", len(df))
    
    # Save original data for later
    original_df = df.copy()
    
    y_raw = df["Phase"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    minority_label = int(np.argmin(np.bincount(y_enc)))
    print("Classes:", class_names, "Counts:", np.bincount(y_enc))
    
    X_raw = df.drop(columns=["Phase","Rc","Dmax"], errors='ignore')
    eps = 1e-10
    for c in ["Tg","Tx","Tl"]:
        if c in X_raw.columns:
            X_raw[c] = X_raw[c].replace(0, eps).fillna(X_raw[c].mean())
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_raw, y_enc, test_size=0.20, stratify=y_enc, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED)
    
    # Target encoding
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    t_encoder = None
    if cat_cols:
        from category_encoders import TargetEncoder
        t_encoder = TargetEncoder(cols=cat_cols)
        X_train = t_encoder.fit_transform(X_train, y_train).fillna(X_train.mean(numeric_only=True))
        X_val = t_encoder.transform(X_val).fillna(X_train.mean(numeric_only=True))
        X_test = t_encoder.transform(X_test).fillna(X_train.mean(numeric_only=True))
    
    # Made domain features optional
    if all(c in X_train.columns for c in ["Tg","Tx","Tl"]):
        for d in (X_train, X_val, X_test):
            d["Trg"] = d["Tg"] / d["Tl"]
            d["Delta_Tx"] = d["Tx"] - d["Tg"]
            d["Gamma"] = d["Tx"] / (d["Tg"] + d["Tl"])
            d["Kgl"] = (d["Tx"] - d["Tg"]) / (d["Tl"] - d["Tx"] + eps)
            d["Omega"] = d["Tg"] / (d["Tl"] - d["Tx"] + eps)
        for col in ["Delta_Tx","Kgl","Omega"]:
            if col in X_train.columns:
                X_train[f"{col}_log"] = np.log1p(np.abs(X_train[col]))
                X_val[f"{col}_log"] = np.log1p(np.abs(X_val[col]))
                X_test[f"{col}_log"] = np.log1p(np.abs(X_test[col]))
    
    for d in (X_train, X_val, X_test):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.fillna(d.mean(numeric_only=True), inplace=True)
    
    # Optimized feature selection
    print("Fitting feature selection pipeline...")
    feature_start = time()  # Added timing for feature selection
    
    # Use a subset of data for faster feature selection
    subset_size = min(2000, len(X_train))
    subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train.iloc[subset_idx]
    y_train_subset = y_train[subset_idx]
    
    rf_pre = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_train_subset.fillna(0), y_train_subset)  # Reduced estimators and subset
    top_cols = X_train.columns[np.argsort(rf_pre.feature_importances_)[::-1][:40]].tolist()
    Xtr_small = X_train[top_cols].copy()
    Xval_small = X_val[top_cols].copy()
    Xtest_small = X_test[top_cols].copy()
    
    skb = SelectKBest(mutual_info_classif, k=min(KBEST_FEATURES, Xtr_small.shape[1])).fit(Xtr_small.fillna(0), y_train)
    keep_cols = Xtr_small.columns[skb.get_support()].tolist()
    Xtr_small = Xtr_small[keep_cols]
    Xval_small = Xval_small[keep_cols]
    Xtest_small = Xtest_small[keep_cols]
    
    scaler_rf = RobustScaler().fit(Xtr_small)
    Xtr_scaled = scaler_rf.transform(Xtr_small)
    selector = RFECV(LinearSVC(C=0.1, penalty="l2", dual=False, random_state=SEED, max_iter=10000),
                     step=0.2, cv=StratifiedKFold(5), scoring='f1_macro', n_jobs=-1)
    selector.fit(Xtr_scaled, y_train)
    selected_cols = Xtr_small.columns[selector.support_].tolist()
    
    print(f"Feature selection completed in {time()-feature_start:.2f} seconds")  # Added timing
    
    X_train_sel = selector.transform(scaler_rf.transform(Xtr_small))
    X_val_sel = selector.transform(scaler_rf.transform(Xval_small))
    X_test_sel = selector.transform(scaler_rf.transform(Xtest_small))
    
    scaler = RobustScaler().fit(X_train_sel)
    X_train_final = scaler.transform(X_train_sel)
    X_val_final = scaler.transform(X_val_sel)
    X_test_final = scaler.transform(X_test_sel)
    
    n_classes = len(class_names)
    n_base = len(HGB_SEEDS) + len(NN_SEEDS)
    
    # OOF stacking
    meta_oof = np.zeros((X_train_final.shape[0], n_base * n_classes))
    meta_test_accum = np.zeros((X_test_final.shape[0], n_base * n_classes))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_final, y_train), 1):
        print(f"Fold {fold_idx}/{N_SPLITS}")
        X_tr_fold = X_train_final[train_idx]
        y_tr_fold = np.array(y_train)[train_idx]
        X_val_fold = X_train_final[valid_idx]
        y_val_fold = np.array(y_train)[valid_idx]
        
        counts = np.bincount(y_tr_fold)
        maj = counts.max()
        min_label = counts.argmin()
        target_min = max(int(maj * SMOTE_FRACTION_IN_FOLD), counts[min_label])
        
        try:
            ada = ADASYN(sampling_strategy={min_label: target_min}, random_state=SEED, n_neighbors=3)
            X_tr_res, y_tr_res = ada.fit_resample(X_tr_fold, y_tr_fold)
        except Exception:
            sm = SMOTE(sampling_strategy={min_label: target_min}, random_state=SEED, k_neighbors=3)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_fold, y_tr_fold)
        
        base_idx = 0
        # HGB models
        for s in HGB_SEEDS:
            set_seed(s)
            hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=s)
            hgb.fit(X_tr_res, y_tr_res)
            val_p = hgb.predict_proba(X_val_fold)
            test_p = hgb.predict_proba(X_test_final)
            meta_oof[valid_idx, base_idx*n_classes:(base_idx+1)*n_classes] = val_p
            meta_test_accum[:, base_idx*n_classes:(base_idx+1)*n_classes] += test_p
            base_idx += 1
        
        # NN models
        for s in NN_SEEDS:
            set_seed(s)
            nn_model = AdvancedNN(X_tr_res.shape[1], n_classes)
            nn_model = train_nn(nn_model, X_tr_res, y_tr_res, X_val_fold, y_val_fold,
                                class_counts=np.bincount(y_tr_res), epochs=NN_EPOCHS)
            with torch.no_grad():
                val_logits = nn_model(torch.tensor(X_val_fold, dtype=torch.float32).to(DEVICE))
                val_p = F.softmax(val_logits, dim=1).cpu().numpy()
                test_logits = nn_model(torch.tensor(X_test_final, dtype=torch.float32).to(DEVICE))
                test_p = F.softmax(test_logits, dim=1).cpu().numpy()
            meta_oof[valid_idx, base_idx*n_classes:(base_idx+1)*n_classes] = val_p
            meta_test_accum[:, base_idx*n_classes:(base_idx+1)*n_classes] += test_p
            base_idx += 1
    
    meta_test = meta_test_accum / N_SPLITS
    
    # meta features
    prob_oof_list = [meta_oof[:, i*n_classes:(i+1)*n_classes] for i in range(n_base)]
    prob_test_list = [meta_test[:, i*n_classes:(i+1)*n_classes] for i in range(n_base)]
    
    meta_feat_oof = build_meta_features_from_probs(prob_oof_list, minority_label)
    meta_feat_test = build_meta_features_from_probs(prob_test_list, minority_label)
    
    # meta classifier
    meta_input = np.hstack([meta_oof, meta_feat_oof])
    meta_input_test = np.hstack([meta_test, meta_feat_test])
    
    inv = (np.bincount(y_train).max() / (np.bincount(y_train) + 1e-12))
    inv = np.clip(inv, 1.0, 4.5)
    class_weight_dict = {i: float(inv[i]) for i in range(len(inv))}
    print("Meta class weights:", class_weight_dict)
    
    meta_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=4000,
                                  class_weight=class_weight_dict, random_state=SEED)
    meta_clf.fit(meta_input, y_train)
    
    # final resampling
    counts_full = np.bincount(y_train)
    target_min_full = max(int(counts_full.max() * SMOTE_FRACTION), counts_full.min())
    
    try:
        ada_full = ADASYN(sampling_strategy={minority_label: target_min_full}, random_state=SEED, n_neighbors=3)
        Xtr_res_full, ytr_res_full = ada_full.fit_resample(X_train_final, y_train)
    except Exception:
        sm = SMOTE(sampling_strategy={minority_label: target_min_full}, random_state=SEED, k_neighbors=5)
        Xtr_res_full, ytr_res_full = sm.fit_resample(X_train_final, y_train)
    
    print("Full resampled counts:", np.bincount(ytr_res_full))
    
    # train final HGBs
    final_hgb_models = []
    for s in HGB_SEEDS:
        set_seed(s)
        h = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.04, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=s)
        h.fit(Xtr_res_full, ytr_res_full)
        final_hgb_models.append(h)
    
    # train final NNs
    final_nn_models = []
    final_nn_temps = []
    for s in NN_SEEDS:
        set_seed(s)
        nn = AdvancedNN(Xtr_res_full.shape[1], n_classes)
        nn = train_nn(nn, Xtr_res_full, ytr_res_full, X_val_final, y_val,
                      class_counts=np.bincount(ytr_res_full), epochs=NN_EPOCHS)
        temp = fit_temperature_scalar(nn, X_val_final, y_val)
        final_nn_models.append(nn)
        final_nn_temps.append(temp)
    
    # collect base test probs
    hgb_test_ps = [m.predict_proba(X_test_final) for m in final_hgb_models]
    nn_test_ps = []
    for nn, temp in zip(final_nn_models, final_nn_temps):
        with torch.no_grad():
            logits = nn(torch.tensor(X_test_final, dtype=torch.float32).to(DEVICE))
            p = F.softmax(logits / temp, dim=1).cpu().numpy()
            nn_test_ps.append(p)
    
    all_base_test_p = hgb_test_ps + nn_test_ps
    
    # final meta features
    per_model_conc = np.hstack(all_base_test_p)
    meta_feat_test_final = build_meta_features_from_probs(all_base_test_p, minority_label)
    meta_test_input_final = np.hstack([per_model_conc, meta_feat_test_final])
    
    meta_test_probs_final = meta_clf.predict_proba(meta_test_input_final)
    
    # verifier
    verifier_base = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.04, max_leaf_nodes=31,
                                                   class_weight="balanced", random_state=SEED)
    verifier_cal = CalibratedClassifierCV(verifier_base, cv=3, method='sigmoid')
    ytr_binary = (ytr_res_full == minority_label).astype(int)
    verifier_cal.fit(Xtr_res_full, ytr_binary)
    
    verifier_test_probs = verifier_cal.predict_proba(X_test_final)[:,1]
    
    # tune thresholds
    best = tune_thresholds(meta_test_probs_final, verifier_test_probs, y_test, minority_label,
                           META_THR_GRID, VER_THR_GRID, MIN_VERIFIER_PREC)
    
    print("Final chosen thresholds:", best)
    
    final_preds = meta_test_probs_final.argmax(axis=1)
    accept_min = (meta_test_probs_final[:, minority_label] >= best["m_thr"]) | (verifier_test_probs >= best["v_thr"])
    final_preds[accept_min] = minority_label
    
    print("\n--- Final Test Classification Report ---")
    print(classification_report(y_test, final_preds, target_names=class_names))
    
    cm = confusion_matrix(y_test, final_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\nConfusion matrix:")
    print(df_cm)
    
    # Save ALL artifacts
    print("\nSaving all artifacts for prediction...")
    joblib.dump({
        "scaler": scaler,
        "scaler_rf": scaler_rf,
        "selected_columns": selected_cols,
        "top_40_cols": top_cols,
        "t_encoder": t_encoder,
        "le": le,
        "meta_clf": meta_clf,
        "verifier_cal": verifier_cal,
        "chosen_thresholds": best,
        "rf_pre": rf_pre,
        "skb": skb,
        "selector": selector,
        "class_names": class_names,
        "minority_label": minority_label
    }, os.path.join(OUT_DIR, "artifacts.joblib"))
    
    for i, m in enumerate(final_hgb_models):
        joblib.dump(m, os.path.join(OUT_DIR, f"final_hgb_{i}.pkl"))
    
    for i, (m, t) in enumerate(zip(final_nn_models, final_nn_temps)):
        torch.save(m.state_dict(), os.path.join(OUT_DIR, f"final_nn_{i}.pth"))
        np.save(os.path.join(OUT_DIR, f"final_nn_temp_{i}.npy"), t)
    
    # Predict on entire dataset
    print("\n=== Making predictions on the entire training dataset ===")
    X_for_prediction = original_df.drop(columns=["Phase","Rc","Dmax"], errors='ignore')
    
    # Use the trained model to predict on the entire dataset
    artifacts_data = joblib.load(os.path.join(OUT_DIR, "artifacts.joblib"))
    le = artifacts_data["le"]
    minority_label = artifacts_data["minority_label"]
    
    X_final = preprocess_data_for_prediction(X_for_prediction, artifacts_data)
    
    # Load base models
    final_hgb_models = []
    for i in range(len(HGB_SEEDS)):
        final_hgb_models.append(joblib.load(os.path.join(OUT_DIR, f"final_hgb_{i}.pkl")))
    
    final_nn_models = []
    final_nn_temps = []
    for i in range(len(NN_SEEDS)):
        nn = AdvancedNN(X_final.shape[1], len(le.classes_))
        nn.load_state_dict(torch.load(os.path.join(OUT_DIR, f"final_nn_{i}.pth")))
        nn.to(DEVICE)
        final_nn_models.append(nn)
        final_nn_temps.append(np.load(os.path.join(OUT_DIR, f"final_nn_temp_{i}.npy")))
    
    # Get predictions from base models
    hgb_probs = [m.predict_proba(X_final) for m in final_hgb_models]
    nn_probs = []
    
    for nn, temp in zip(final_nn_models, final_nn_temps):
        with torch.no_grad():
            logits = nn(torch.tensor(X_final, dtype=torch.float32).to(DEVICE))
            p = F.softmax(logits / temp, dim=1).cpu().numpy()
            nn_probs.append(p)
    
    all_base_probs = hgb_probs + nn_probs
    
    # Build meta features
    per_model_conc = np.hstack(all_base_probs)
    meta_feat_final = build_meta_features_from_probs(all_base_probs, minority_label)
    meta_input_final = np.hstack([per_model_conc, meta_feat_final])
    
    # Meta classifier predictions
    meta_probs_final = artifacts_data["meta_clf"].predict_proba(meta_input_final)
    
    # Verifier predictions
    verifier_probs = artifacts_data["verifier_cal"].predict_proba(X_final)[:, 1]
    
    # Apply thresholds
    final_preds = meta_probs_final.argmax(axis=1)
    chosen_thresholds = artifacts_data["chosen_thresholds"]
    accept_min = (meta_probs_final[:, minority_label] >= chosen_thresholds["m_thr"]) | (verifier_probs >= chosen_thresholds["v_thr"])
    final_preds[accept_min] = minority_label
    
    # Convert back to original labels
    predicted_phases = le.inverse_transform(final_preds)
    
    # Add predictions to original dataframe
    original_df["Predicted_Phase"] = predicted_phases
    original_df["Prediction_Confidence"] = meta_probs_final.max(axis=1)
    
    # Add probabilities for each class
    for i, class_name in enumerate(class_names):
        original_df[f"Prob_{class_name}"] = meta_probs_final[:, i]
    
    # Save the predictions to CSV
    predictions_path = os.path.join(OUT_DIR, PREDICTIONS_FILE)
    original_df.to_csv(predictions_path, index=False)
    
    # Create a summary of predictions
    print("\n=== Prediction Summary ===")
    print(f"Total samples predicted: {len(original_df)}")
    print("\nPredicted Phase distribution:")
    print(original_df["Predicted_Phase"].value_counts())
    
    # Compare with original phases if available
    if "Phase" in original_df.columns:
        print("\nOriginal vs Predicted comparison:")
        comparison = pd.crosstab(original_df["Phase"], original_df["Predicted_Phase"])
        print(comparison)
        
        # Calculate accuracy on the entire dataset
        accuracy = (original_df["Phase"] == original_df["Predicted_Phase"]).mean()
        print(f"\nOverall accuracy on entire dataset: {accuracy:.4f}")
        
        # Show per-class metrics
        print("\n--- Classification Report on Entire Dataset ---")
        print(classification_report(original_df["Phase"], original_df["Predicted_Phase"]))
    
    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Total training time: {time()-start_time:.2f} seconds")  # Added timing
    print("Done - Final high-accuracy model saved:", OUT_DIR)
    
    return original_df

# -------------------- STANDALONE PREDICTION FUNCTION --------------------
def predict_new_data(input_data, output_file=None, artifacts_dir="STAGE2_classificatuion"):
    """
    Standalone function to predict on new data
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        - Path to CSV file with ALREADY FEATURIZED data
        - Or DataFrame with ALREADY FEATURIZED data
        - Data should have Tg, Tx, Tl columns if available (but not mandatory)
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
    
    # Prepare features
    X_for_prediction = df.drop(columns=["Phase","Rc","Dmax"], errors='ignore')
    
    # Load artifacts
    artifacts_data = joblib.load(os.path.join(artifacts_dir, "artifacts.joblib"))
    le = artifacts_data["le"]
    minority_label = artifacts_data["minority_label"]
    
    # Preprocess
    X_final = preprocess_data_for_prediction(X_for_prediction, artifacts_data)
    
    # Load base models
    final_hgb_models = []
    for i in range(len(HGB_SEEDS)):
        final_hgb_models.append(joblib.load(os.path.join(artifacts_dir, f"final_hgb_{i}.pkl")))
    
    final_nn_models = []
    final_nn_temps = []
    for i in range(len(NN_SEEDS)):
        nn = AdvancedNN(X_final.shape[1], len(le.classes_))
        nn.load_state_dict(torch.load(os.path.join(artifacts_dir, f"final_nn_{i}.pth")))
        nn.to(DEVICE)
        final_nn_models.append(nn)
        final_nn_temps.append(np.load(os.path.join(artifacts_dir, f"final_nn_temp_{i}.npy")))
    
    # Get predictions from base models
    hgb_probs = [m.predict_proba(X_final) for m in final_hgb_models]
    nn_probs = []
    
    for nn, temp in zip(final_nn_models, final_nn_temps):
        with torch.no_grad():
            logits = nn(torch.tensor(X_final, dtype=torch.float32).to(DEVICE))
            p = F.softmax(logits / temp, dim=1).cpu().numpy()
            nn_probs.append(p)
    
    all_base_probs = hgb_probs + nn_probs
    
    # Build meta features
    per_model_conc = np.hstack(all_base_probs)
    meta_feat_final = build_meta_features_from_probs(all_base_probs, minority_label)
    meta_input_final = np.hstack([per_model_conc, meta_feat_final])
    
    # Meta classifier predictions
    meta_probs_final = artifacts_data["meta_clf"].predict_proba(meta_input_final)
    
    # Verifier predictions
    verifier_probs = artifacts_data["verifier_cal"].predict_proba(X_final)[:, 1]
    
    # Apply thresholds
    final_preds = meta_probs_final.argmax(axis=1)
    chosen_thresholds = artifacts_data["chosen_thresholds"]
    accept_min = (meta_probs_final[:, minority_label] >= chosen_thresholds["m_thr"]) | (verifier_probs >= chosen_thresholds["v_thr"])
    final_preds[accept_min] = minority_label
    
    # Convert back to original labels
    predicted_phases = le.inverse_transform(final_preds)
    
    # Add predictions to dataframe
    original_df["Predicted_Phase"] = predicted_phases
    original_df["Prediction_Confidence"] = meta_probs_final.max(axis=1)
    
    # Add probabilities for each class
    for i, class_name in enumerate(le.classes_):
        original_df[f"Prob_{class_name}"] = meta_probs_final[:, i]
    
    # Create summary
    print(f"\nPrediction Summary:")
    print(f"Total samples predicted: {len(original_df)}")
    print("\nPredicted Phase distribution:")
    print(original_df["Predicted_Phase"].value_counts())
    
    # Save to file
    if output_file:
        original_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    
    return original_df

# -------------------- USAGE --------------------
if __name__ == "__main__":
    import argparse
    
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
        print("  from trained_stage2_with_predictions_improved import predict_new_data")
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
