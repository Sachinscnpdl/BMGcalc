#!/usr/bin/env python3
# unified_predictor.py  (UPDATED)
"""
Unified predictor with robust featurization ordering (fixes single-row vs batch mismatch).
- Ensures featurized outputs always include STAGE1_TOP features in the exact order.
- Downstream-wrapper remains: creates module-specific CSVs and calls predict_new_data.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import joblib
import torch
from typing import Tuple, List, Optional, Dict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Artifact defaults (must exist or make module importable)
DEFAULT_STAGE1_TOP = "stage1_top_features.npy"
DEFAULT_STAGE1_SCALER = "stage1_scaler.joblib"
DEFAULT_STAGE1_MODEL = "stage1_regression_model.pth"
DEFAULT_FEATURIZED_CSV = "featurized_metallic_glass_stage1.csv"

# Debug flag (set env UNIFIED_PREDICTOR_DEBUG=1 to enable extra prints)
DEBUG = bool(os.environ.get("UNIFIED_PREDICTOR_DEBUG", "") != "")

# -------------------------
# Utilities
# -------------------------
def _import_stage3(module_name: str = "stage3_thermal_properties") -> Dict:
    try:
        stage3 = __import__(module_name, fromlist=[
            "STAGE1_TOP", "STAGE1_SCALER", "STAGE1_MODEL", "GlassThermalModel", "DEVICE"
        ])
        return {"module": stage3}
    except Exception:
        files = {
            "STAGE1_TOP": DEFAULT_STAGE1_TOP,
            "STAGE1_SCALER": DEFAULT_STAGE1_SCALER,
            "STAGE1_MODEL": DEFAULT_STAGE1_MODEL
        }
        missing = [v for v in files.values() if not os.path.exists(v)]
        if DEBUG:
            print("stage3 import failed; fallback file map:", files, "missing:", missing)
        return {"module": None, **files}

def _get_required_features() -> Optional[List[str]]:
    """
    Try to obtain the canonical feature list (STAGE1_TOP) used for training.
    Priority:
      1) stage1_top_features.npy in cwd
      2) header of featurized_metallic_glass_stage1.csv (if present)
    Returns list[str] or None.
    """
    if os.path.exists(DEFAULT_STAGE1_TOP):
        try:
            arr = np.load(DEFAULT_STAGE1_TOP, allow_pickle=True)
            return [str(x) for x in arr.tolist()]
        except Exception:
            pass
    if os.path.exists(DEFAULT_FEATURIZED_CSV):
        try:
            df_head = pd.read_csv(DEFAULT_FEATURIZED_CSV, nrows=0)
            return list(df_head.columns)
        except Exception:
            pass
    return None

# -------------------------
# Featurization: always finalize to canonical feature order
# -------------------------
def featurize_alloys_complete(df: pd.DataFrame, formula_col: str = "Alloys") -> pd.DataFrame:
    if formula_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain column '{formula_col}'")
    # Try the real featurizer if available
    try:
        import featurization_module  # type: ignore
        from featurization_module import StrToComposition, MATMINER_AVAILABLE  # type: ignore
        if DEBUG: print("Using featurization_module")
        r = df.copy()
        if hasattr(featurization_module, "StrToComposition"):
            stc = StrToComposition(target_col_id="composition_obj")
            r = stc.featurize_dataframe(r, formula_col, ignore_errors=True)
        else:
            r["composition_obj"] = None

        if getattr(featurization_module, "MATMINER_AVAILABLE", False):
            try:
                _EP_base = getattr(featurization_module, "_EP_base", None)
                _ST_base = getattr(featurization_module, "_ST_base", None)
                _VO_base = getattr(featurization_module, "_VO_base", None)
                _IP_base = getattr(featurization_module, "_IP_base", None)
                featurizers = []
                if _EP_base is not None:
                    featurizers.append(_EP_base.from_preset("magpie"))
                if _ST_base is not None:
                    featurizers.append(_ST_base())
                if _VO_base is not None:
                    featurizers.append(_VO_base())
                if _IP_base is not None:
                    featurizers.append(_IP_base(fast=True))
                for f in featurizers:
                    r = f.featurize_dataframe(r, "composition_obj", ignore_errors=True)
            except Exception:
                if DEBUG: print("matminer featurizers failed; continuing")

        if hasattr(featurization_module, "compute_all_custom_features"):
            custom = r["composition_obj"].apply(featurization_module.compute_all_custom_features)
            r = pd.concat([r, custom], axis=1)

        # drop helpers
        r.drop(columns=[formula_col, "composition_obj"], errors="ignore", inplace=True)

        # try to drop constant/nan columns only if function available (but *do not* rely solely on it)
        if hasattr(featurization_module, "drop_constant_or_nan_columns") and len(r) > 1:
            try:
                r, _ = featurization_module.drop_constant_or_nan_columns(r)
            except Exception:
                if DEBUG: print("drop_constant_or_nan_columns error; skipping")

        # Fill NA numeric->0, categorical->mode
        for col in r.columns:
            if r[col].isna().any():
                if pd.api.types.is_numeric_dtype(r[col]):
                    r[col] = r[col].fillna(0.0)
                else:
                    m = r[col].mode()
                    r[col] = r[col].fillna(m[0] if len(m) else "unknown")
    except Exception as e:
        # fallback minimal featurizer
        if DEBUG: print("featurization_module not used; fallback. exc:", e)
        r = df.copy()
        for col in ["Tg", "Tx", "Tl"]:
            if col not in r.columns:
                r[col] = np.nan
        # try to recover a required feature list
        required = _get_required_features()
        if required is not None:
            for feat in required:
                if feat not in r.columns:
                    r[feat] = 0.0
            # reorder so required features come first (canonical order)
            common = [f for f in required if f in r.columns]
            r = r.reindex(columns=common + [c for c in r.columns if c not in common])
        else:
            placeholders = ["avg_atomic_number", "avg_atomic_mass", "electronegativity_diff"]
            for p in placeholders:
                if p not in r.columns:
                    r[p] = 0.0
        for col in r.columns:
            if pd.api.types.is_numeric_dtype(r[col]):
                r[col] = r[col].fillna(0.0)
            else:
                r[col] = pd.to_numeric(r[col], errors="coerce").fillna(0.0)

    # --- CRITICAL FINALIZATION: enforce canonical features & ordering (fixes single-row vs multi-row mismatch)
    canonical = _get_required_features()
    if canonical is not None:
        # ensure all canonical columns exist, filled numeric
        for feat in canonical:
            if feat not in r.columns:
                r[feat] = 0.0
            else:
                # coerce to numeric if possible and fill na
                r[feat] = pd.to_numeric(r[feat], errors='coerce').fillna(0.0)
        # Reorder: canonical features first (in exact order), then other columns (preserve them after)
        other_cols = [c for c in r.columns if c not in canonical]
        r = r.loc[:, canonical + other_cols]
        if DEBUG:
            print(f"featurize_alloys_complete: enforced canonical features count={len(canonical)}")
    else:
        if DEBUG:
            print("featurize_alloys_complete: canonical feature list not found; returning featurized output as-is")

    return r

# -------------------------
# Stage3 alignment & prediction (unchanged logic, uses canonical features)
# -------------------------
def _load_stage3_top_and_scaler(stage3_info: dict) -> Tuple[List[str], object, Optional[object]]:
    if stage3_info.get("module") is not None:
        mod = stage3_info["module"]
        top_raw = np.load(mod.STAGE1_TOP, allow_pickle=True)
        top = [str(x) for x in top_raw.tolist()]
        scaler = joblib.load(mod.STAGE1_SCALER)
        return top, scaler, mod
    else:
        if os.path.exists(DEFAULT_STAGE1_TOP):
            top_raw = np.load(DEFAULT_STAGE1_TOP, allow_pickle=True)
            top = [str(x) for x in top_raw.tolist()]
        else:
            raise FileNotFoundError("stage1_top_features.npy not found in cwd.")
        if not os.path.exists(DEFAULT_STAGE1_SCALER):
            raise FileNotFoundError("stage1_scaler.joblib not found in cwd.")
        scaler = joblib.load(DEFAULT_STAGE1_SCALER)
        return top, scaler, None

def _align_and_scale(df_in: pd.DataFrame, top_features: List[str], scaler) -> Tuple[np.ndarray, pd.DataFrame]:
    df_local = df_in.copy()
    for feat in top_features:
        if feat not in df_local.columns:
            df_local[feat] = 0.0
    df_ordered = df_local.loc[:, top_features]
    for c in df_ordered.columns:
        df_ordered[c] = pd.to_numeric(df_ordered[c], errors='coerce').fillna(0.0)
    X = df_ordered.values.astype(np.float32)
    if hasattr(scaler, "n_features_in_"):
        expected = int(scaler.n_features_in_)
        if expected != X.shape[1]:
            raise ValueError(f"Scaler expects {expected} features but provided {X.shape[1]}.")
    X_scaled = scaler.transform(X)
    return X_scaled, df_ordered

def predict_thermal_for_alloys(featurized_path: str, stage3_module_name: str = "stage3_thermal_properties", batch_size: int = 64) -> pd.DataFrame:
    if not os.path.exists(featurized_path):
        raise FileNotFoundError(f"Featurized file not found: {featurized_path}")
    stage3_info = _import_stage3(stage3_module_name)
    top_features, scaler, stage3_mod = _load_stage3_top_and_scaler(stage3_info)
    df = pd.read_csv(featurized_path)
    # Align and scale using canonical top_features
    X_scaled, df_ordered = _align_and_scale(df, top_features, scaler)
    # Load model class and state
    if stage3_info.get("module") is not None:
        mod = stage3_info["module"]
        GlassThermalModel = mod.GlassThermalModel
        DEVICE = getattr(mod, "DEVICE", torch.device("cpu"))
        model_path = mod.STAGE1_MODEL
    else:
        mod = __import__(stage3_module_name, fromlist=['GlassThermalModel', 'DEVICE'])
        GlassThermalModel = mod.GlassThermalModel
        DEVICE = getattr(mod, "DEVICE", torch.device("cpu"))
        model_path = mod.STAGE1_MODEL
    model = GlassThermalModel(input_dim=len(top_features))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stage3 model file not found at: {model_path}")
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_scaled.shape[0], batch_size):
            batch = torch.from_numpy(X_scaled[i:i+batch_size]).float().to(DEVICE)
            out = model(batch).cpu().numpy()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)
    result = df.copy()
    result["Predicted_Tg"] = preds[:, 0]
    result["Predicted_Tx"] = preds[:, 1]
    result["Predicted_Tl"] = preds[:, 2]
    if DEBUG:
        print("predict_thermal_for_alloys: predicted shape", preds.shape)
    return result

# -------------------------
# Downstream compatibility wrapper helpers (unchanged)
# -------------------------
def _discover_module_top_features(module, module_name_hint: str) -> Optional[List[str]]:
    for attr in dir(module):
        if attr.isupper() and 'TOP' in attr:
            val = getattr(module, attr)
            try:
                arr = np.array(val)
                return [str(x) for x in arr.tolist()]
            except Exception:
                continue
    for attr in dir(module):
        if 'top' in attr.lower():
            try:
                val = getattr(module, attr)
                arr = np.array(val)
                return [str(x) for x in arr.tolist()]
            except Exception:
                continue
    candidates = [
        f"{module_name_hint}_top_features.npy",
        f"{module.__name__}_top_features.npy",
        "stage2_top_features.npy",
        "stage4_top_features.npy",
        "stage5_top_features.npy",
        "top_features.npy"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                arr = np.load(path, allow_pickle=True)
                return [str(x) for x in arr.tolist()]
            except Exception:
                continue
    return None

def _make_module_input_csv_for(module_name: str, df: pd.DataFrame, temp_dir: str) -> Tuple[str, Optional[List[str]]]:
    try:
        mod = __import__(module_name, fromlist=['*'])
    except Exception:
        mod = None
    top = None
    if mod is not None:
        top = _discover_module_top_features(mod, module_name)
    else:
        candidates = [
            f"{module_name}_top_features.npy",
            f"{module_name.split('.')[-1]}_top_features.npy",
            "stage2_top_features.npy",
            "stage4_top_features.npy",
            "stage5_top_features.npy",
            "top_features.npy"
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    arr = np.load(c, allow_pickle=True)
                    top = [str(x) for x in arr.tolist()]
                    break
                except Exception:
                    continue
    if top is None:
        fallback_path = os.path.join(temp_dir, f"{module_name.replace('.', '_')}_input_fallback.csv")
        df_numeric = df.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0.0)
        df_numeric.to_csv(fallback_path, index=False)
        if DEBUG:
            print(f"_make_module_input_csv_for: fallback created for {module_name} with {df_numeric.shape[1]} columns")
        return fallback_path, None
    df_mod = df.copy()
    for feat in top:
        if feat not in df_mod.columns:
            df_mod[feat] = 0.0
    df_top = df_mod.loc[:, top]
    for c in df_top.columns:
        df_top[c] = pd.to_numeric(df_top[c], errors='coerce').fillna(0.0)
    out_path = os.path.join(temp_dir, f"{module_name.replace('.', '_')}_input.csv")
    df_top.to_csv(out_path, index=False)
    if DEBUG:
        print(f"_make_module_input_csv_for: created {out_path} with {len(top)} features for {module_name}")
    return out_path, top

def _safe_call_module_predict(module_name: str, featurized_df: pd.DataFrame, func_name: str = "predict_new_data"):
    temp_dir = os.path.join(os.getcwd(), "prediction_temp")
    os.makedirs(temp_dir, exist_ok=True)
    module_input_csv, used_top = _make_module_input_csv_for(module_name, featurized_df, temp_dir)
    if DEBUG:
        print(f"[downstream-wrapper] Calling {module_name}.{func_name} with input CSV: {module_input_csv} (features_used={None if used_top is None else len(used_top)})")
    try:
        mod = __import__(module_name, fromlist=[func_name])
    except Exception as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    if not hasattr(mod, func_name):
        raise AttributeError(f"Module '{module_name}' does not expose function '{func_name}'")
    predict_fn = getattr(mod, func_name)
    res = predict_fn(module_input_csv)
    return res

# -------------------------
# Main pipeline (same usage)
# -------------------------
def process_alloys(input_data, output_file: str = None, stage3_module_name: str = "stage3_thermal_properties"):
    if isinstance(input_data, str):
        if input_data.endswith(".csv") and os.path.exists(input_data):
            df_input = pd.read_csv(input_data)
            input_base = os.path.splitext(os.path.basename(input_data))[0]
            print(f"Loaded {len(df_input)} rows from CSV: {input_data}")
        else:
            df_input = pd.DataFrame({"Alloys": [input_data]})
            input_base = "single_alloy"
            print(f"Processing single alloy: {input_data}")
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        input_base = "dataframe_input"
    else:
        raise ValueError("input_data must be CSV path, DataFrame, or a single alloy string.")
    temp_dir = os.path.join(os.getcwd(), "prediction_temp")
    os.makedirs(temp_dir, exist_ok=True)
    featurized_path = os.path.join(temp_dir, f"{input_base}_featurized.csv")
    print("Featurizing (or reconstructing) features...")
    featurized_df = featurize_alloys_complete(df_input, formula_col="Alloys")
    for col in ["Tg", "Tx", "Tl"]:
        if col in df_input.columns:
            featurized_df[col] = df_input[col].values
            if DEBUG: print(f"Preserved experimental {col} from input")
    featurized_df.to_csv(featurized_path, index=False)
    print(f"Featurized file saved to: {featurized_path}")
    print("Predicting thermal properties with Stage 3 model...")
    thermal_predictions = predict_thermal_for_alloys(featurized_path, stage3_module_name=stage3_module_name)
    merged = featurized_df.copy()
    for col, pred_col in zip(["Tg", "Tx", "Tl"], ["Predicted_Tg", "Predicted_Tx", "Predicted_Tl"]):
        merged[col] = pd.to_numeric(merged.get(col, np.nan), errors='coerce')
        mask_fill = merged[col].isna() | (merged[col] <= 0)
        filled_count = mask_fill.sum()
        merged.loc[mask_fill, col] = thermal_predictions.loc[mask_fill, pred_col].values
        print(f"Filled {filled_count} rows for {col} using stage3 predictions.")
    merged["delta_Tx"] = merged["Tx"] - merged["Tg"]
    merged["delta_Tx"] = np.maximum(merged["delta_Tx"].fillna(25.0), 25.0)
    merged["Trg"] = np.where(merged["Tl"] > 0, merged["Tg"] / merged["Tl"], 0.0)
    denom = merged["Tg"].fillna(0.0) + merged["Tl"].fillna(0.0)
    merged["gamma"] = np.where(denom > 0, merged["Tx"] / denom, 0.0)
    merged.to_csv(featurized_path, index=False)
    print(f"Merged featurized data saved to: {featurized_path}")
    final_results = df_input.reset_index(drop=True).copy()
    # Phase
    try:
        phase_df = _safe_call_module_predict("trained_stage2_with_predictions_flexible", merged)
        final_results["Predicted_Phase"] = phase_df.get("Predicted_Phase").values
    except Exception as e:
        print("Phase prediction step failed:", e)
        final_results["Predicted_Phase"] = None
    # Dmax
    try:
        dmax_df = _safe_call_module_predict("stage4_dmax", merged)
        final_results["Predicted_Dmax"] = dmax_df.get("Predicted_Dmax").values
    except Exception as e:
        print("Dmax prediction step failed:", e)
        final_results["Predicted_Dmax"] = None
    # Rc
    try:
        rc_df = _safe_call_module_predict("stage5_rc", merged)
        final_results["Predicted_Rc"] = rc_df.get("Predicted_Rc").values
    except Exception as e:
        print("Rc prediction step failed:", e)
        final_results["Predicted_Rc"] = None
    # Attach thermal predictions
    final_results["Predicted_Tg"] = thermal_predictions["Predicted_Tg"].values
    final_results["Predicted_Tx"] = thermal_predictions["Predicted_Tx"].values
    final_results["Predicted_Tl"] = thermal_predictions["Predicted_Tl"].values
    if output_file:
        final_results.to_csv(output_file, index=False)
        print(f"Final results saved to: {output_file}")
    return final_results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    if args.debug:
        os.environ["UNIFIED_PREDICTOR_DEBUG"] = "1"
    inp = args.input
    if os.path.exists(inp) and inp.endswith(".csv"):
        out = process_alloys(inp, output_file=args.output)
    else:
        out = process_alloys(inp, output_file=args.output)
    print(out.head())
