# Authored by: Sachin Poudel, Silesian University, Poland
import os
import tempfile
import shutil
import re
from typing import Optional, Union

import numpy as np
import pandas as pd

# Your existing modules (these must be importable from your environment)
import featurization_module as feat_mod
import stage3_thermal_properties as s3_mod
import stage4_dmax as s4_mod
import stage5_rc as s5_mod
import trained_stage2_with_predictions_flexible as s2_mod


def featurize_alloys_complete(df, formula_col="Alloys"):
    """
    Complete featurization for alloys that ensures all features match training data.
    This is the conservative fallback to use when the fast featurizer removes all columns
    (e.g., because a single-row had all-constant values dropped).
    """
    import numpy as np
    import pandas as pd

    # Try to load a canonical featurized header to preserve expected columns if available
    try:
        original_df = pd.read_csv("featurized_metallic_glass_stage1.csv")
        required_features = list(original_df.columns)
    except Exception:
        required_features = None

    result_df = df.copy()

    if formula_col not in result_df.columns:
        raise ValueError(f"Column '{formula_col}' not found in DataFrame")

    # Use StrToComposition if available to create composition objects
    try:
        from featurization_module import StrToComposition, MATMINER_AVAILABLE
    except Exception:
        StrToComposition = None
        MATMINER_AVAILABLE = False

    if MATMINER_AVAILABLE and StrToComposition is not None:
        result_df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
            result_df, formula_col, ignore_errors=True
        )
    else:
        result_df["composition_obj"] = None

    # Add matminer-like features if provided by your featurization_module
    if MATMINER_AVAILABLE:
        try:
            from featurization_module import (_EP_base, _ST_base, _VO_base, _IP_base)
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
                result_df = f.featurize_dataframe(result_df, "composition_obj", ignore_errors=True)
        except Exception:
            pass

    # Add custom computed features from your featurization module
    try:
        from featurization_module import compute_all_custom_features
        custom = result_df["composition_obj"].apply(compute_all_custom_features)
        result_df = pd.concat([result_df, custom], axis=1)
    except Exception:
        # If compute_all_custom_features not available or fails, continue
        pass

    # For single rows, avoid aggressive dropping of constant columns
    if len(result_df) > 1:
        try:
            from featurization_module import drop_constant_or_nan_columns
            result_df, _ = drop_constant_or_nan_columns(result_df)
        except Exception:
            pass

    # Drop helper columns
    result_df.drop(columns=["composition_obj", formula_col], inplace=True, errors="ignore")

    # If we have a required feature list, ensure columns exist (fill with NaN)
    if required_features is not None:
        for feature in required_features:
            if feature not in result_df.columns:
                result_df[feature] = np.nan
        # keep ordering to match original (but only columns present)
        common_features = [f for f in required_features if f in result_df.columns]
        result_df = result_df[common_features]

    # Fill NaNs sensibly: numeric -> 0, categorical -> mode or 'unknown'
    for col in result_df.columns:
        if result_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(0)
            else:
                most_frequent = result_df[col].mode()
                if not most_frequent.empty:
                    result_df[col] = result_df[col].fillna(most_frequent[0])
                else:
                    result_df[col] = result_df[col].fillna("unknown")

    return result_df


class ModularBMGPipeline:
    """
    Modular pipeline that supports:
     - CSV path input (with 'Alloys' column)
     - pandas.DataFrame input (with 'Alloys' column)
     - single-alloy string input (e.g., "Zr65Cu15Ni10Al10")
    """
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="bmg_pipe_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _prepare_input_df(self, input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Normalize input to a DataFrame with 'Alloys' column."""
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, str):
            if os.path.exists(input_data) and input_data.lower().endswith(".csv"):
                df = pd.read_csv(input_data)
            else:
                df = pd.DataFrame({"Alloys": [input_data]})
        else:
            raise ValueError("Input must be a CSV path, a pandas DataFrame, or a single alloy string")

        if 'Alloys' not in df.columns:
            raise ValueError("Input data must contain an 'Alloys' column (or be a single alloy string)")

        df['Alloys'] = df['Alloys'].astype(str).reset_index(drop=True)
        return df

    def run_pipeline(self, input_data: Union[str, pd.DataFrame], output_csv: Optional[str] = "final_results.csv"):
        """
        Run full pipeline. If output_csv is None, returns final DataFrame instead of writing.
        """
        print(f"--- Starting pipeline ---")
        df_input = self._prepare_input_df(input_data)
        print(f"Input rows: {len(df_input)}")

        df_result = df_input.copy()

        # ---------------------
        # Step 1: Featurize
        # ---------------------
        print("Step 1: Featurizing compositions...")
        try:
            df_feats = feat_mod.featurize_alloys(df_input)
        except Exception as e_feat:
            print(f"[WARN] Primary featurizer failed: {e_feat!s}")
            df_feats = pd.DataFrame(index=range(len(df_input)))

        df_feats = df_feats.reset_index(drop=True)

        # If featurizer produced ZERO columns, try conservative fallback
        if df_feats.shape[1] == 0:
            print("[WARN] Featurizer returned 0 columns. Trying conservative fallback...")
            try:
                df_feats = featurize_alloys_complete(df_input, formula_col="Alloys").reset_index(drop=True)
                print(f"[INFO] Fallback featurizer returned shape: {df_feats.shape}")
            except Exception as e_fallback:
                print(f"[WARN] Fallback featurizer failed: {e_fallback!s}")

        # If still no columns, add a dummy numeric column to avoid EmptyDataError on CSV read
        if df_feats.shape[1] == 0:
            print("[WARN] No features available after fallback. Adding dummy '__featurizer_dummy__' = 0")
            df_feats = df_feats.copy()
            df_feats["__featurizer_dummy__"] = 0.0

        # Helpful debug info
        #print("Featurized columns:", list(df_feats.columns))
        temp_feat_path = os.path.join(self.temp_dir, "step1_feats.csv")
        df_feats.to_csv(temp_feat_path, index=False)

        # ---------------------
        # Step 2: Stage-3 thermal predictions
        # ---------------------
        print("Step 2: Predicting Thermal Properties (Tg, Tx, Tl)...")

        # Try calling with path, then with DataFrame, then fallback to NaNs
        try:
            thermal_results = s3_mod.predict_new_data(temp_feat_path)
            print("[INFO] Stage-3 predictions produced via file path.")
        except Exception as e_stage3_path:
            print(f"[WARN] stage3.predict_new_data(path) failed: {e_stage3_path!s}")
            try:
                thermal_results = s3_mod.predict_new_data(df_feats)
                print("[INFO] Stage-3 predictions produced via DataFrame input (fallback).")
            except Exception as e_stage3_df:
                print(f"[WARN] stage3.predict_new_data(DataFrame) failed: {e_stage3_df!s}")
                # fallback: create NaN predictions
                n = len(df_feats)
                thermal_results = pd.DataFrame({
                    "Predicted_Tg": [np.nan] * n,
                    "Predicted_Tx": [np.nan] * n,
                    "Predicted_Tl": [np.nan] * n
                })
                print("[ERROR] stage3 unavailable — thermal predictions filled with NaN to continue pipeline.")

        df_thermal = thermal_results.copy().reset_index(drop=True)

        # Normalize predicted columns (support raw names too)
        for col in ['Tg', 'Tx', 'Tl']:
            pred_col = f'Predicted_{col}'
            if pred_col not in df_thermal.columns and col in df_thermal.columns:
                df_thermal[pred_col] = df_thermal[col].values
            if pred_col not in df_thermal.columns:
                # create column of NaNs with correct length
                df_thermal[pred_col] = np.nan

        # Attach predicted thermal columns to df_result with broadcasting safety
        for col in ['Predicted_Tg', 'Predicted_Tx', 'Predicted_Tl']:
            vals = df_thermal.get(col, pd.Series([np.nan] * len(df_thermal))).values
            if len(vals) == len(df_result):
                df_result[col] = vals
            elif len(vals) == 1 and len(df_result) > 1:
                df_result[col] = np.repeat(vals, len(df_result))
            else:
                # If shape mismatch for single-row input, try to broadcast
                df_result[col] = np.nan

        # ---------------------
        # Step 3: Thermodynamic indicators
        # ---------------------
        print("Step 3: Calculating thermodynamic indicators...")
        pred_tg = df_result.get('Predicted_Tg', pd.Series([np.nan] * len(df_result))).values.astype(float)
        pred_tx = df_result.get('Predicted_Tx', pd.Series([np.nan] * len(df_result))).values.astype(float)
        pred_tl = df_result.get('Predicted_Tl', pd.Series([np.nan] * len(df_result))).values.astype(float)

        delta_tx = np.where(np.isnan(pred_tx) | np.isnan(pred_tg), np.nan, pred_tx - pred_tg)
        trg = np.where((~np.isnan(pred_tl)) & (pred_tl != 0), pred_tg / pred_tl, np.nan)
        denom = pred_tg + pred_tl
        gamma = np.where((~np.isnan(denom)) & (denom != 0), pred_tx / denom, np.nan)

        df_result['delta_Tx'] = delta_tx
        df_result['Trg'] = trg
        df_result['gamma'] = gamma

        # ---------------------
        # Step 4: Prepare Stage-2 input
        # ---------------------
        print("Step 4: Preparing Stage-2 input...")
        df_stage2 = df_feats.copy().reset_index(drop=True)

        # Add thermal properties to stage2 input (align lengths)
        df_stage2['Tg'] = pred_tg
        df_stage2['Tx'] = pred_tx
        df_stage2['Tl'] = pred_tl
        df_stage2['delta_Tx'] = delta_tx
        df_stage2['Trg'] = trg
        df_stage2['gamma'] = gamma

        # Align to Stage-2 expected features if present in module
        want_cols = None
        for attr in ['EXPECTED_FEATURES', 'FEATURE_LIST', 'REQUIRED_COLUMNS', 'TOP_FEATURES', 'FEATURES_ORDER']:
            if hasattr(s2_mod, attr):
                val = getattr(s2_mod, attr)
                if isinstance(val, (list, tuple, np.ndarray)):
                    want_cols = [str(x) for x in val]
                elif isinstance(val, str) and os.path.exists(val):
                    try:
                        if val.endswith('.npy'):
                            want_cols = [str(x) for x in np.load(val)]
                        else:
                            want_cols = pd.read_csv(val).columns.tolist()
                    except Exception:
                        want_cols = None
                if want_cols is not None:
                    break

        if want_cols is not None:
            missing = [c for c in want_cols if c not in df_stage2.columns]
            if missing:
                print(f"[WARN] Stage-2 expected list exists; adding {len(missing)} missing columns as zeros")
                for c in missing:
                    df_stage2[c] = 0.0
            df_stage2 = df_stage2.reindex(columns=want_cols)
            print(f"[INFO] df_stage2 aligned to Stage-2 expected list. Shape: {df_stage2.shape}")

        temp_stage2_path = os.path.join(self.temp_dir, "step3_for_stage2.csv")
        df_stage2.to_csv(temp_stage2_path, index=False)

        # ---------------------
        # Step 5: Stage-2 Phase prediction
        # ---------------------
        print("Step 5: Predicting Phase (Stage-2)...")
        try:
            phase_results = s2_mod.predict_new_data(temp_stage2_path)
        except Exception as e_path:
            msg = str(e_path)
            m = re.search(r"expected\s+(\d+)", msg)
            if m:
                expected = int(m.group(1))
                current = df_stage2.shape[1]
                print(f"[Stage-2] Error indicates expected columns = {expected}, current = {current}.")
                if current > expected:
                    trimmed = df_stage2.copy()
                    removed = []
                    cols_order = list(trimmed.columns)
                    for col in reversed(cols_order):
                        if trimmed.shape[1] <= expected:
                            break
                        trimmed.drop(columns=[col], inplace=True)
                        removed.append(col)
                    print(f"[Stage-2] Trimmed {len(removed)} columns to match expected count")
                    trimmed.to_csv(temp_stage2_path, index=False)
                    try:
                        phase_results = s2_mod.predict_new_data(temp_stage2_path)
                    except Exception as e_retry:
                        try:
                            phase_results = s2_mod.predict_new_data(trimmed)
                        except Exception as e_df:
                            raise RuntimeError(f"Stage-2 failed after trimming. Last error: {e_df}") from e_df
                else:
                    raise RuntimeError(f"Stage-2 error when calling with {df_stage2.shape[1]} cols: {e_path}") from e_path
            else:
                try:
                    phase_results = s2_mod.predict_new_data(df_stage2)
                except Exception as e_df:
                    raise RuntimeError(f"Stage-2 failed for both path and DataFrame calls. Path error: {e_path}; DF error: {e_df}") from e_df

        if phase_results is None:
            raise RuntimeError("Stage-2 did not return predictions")

        # Add phase predictions robustly
        if 'Predicted_Phase' in phase_results.columns:
            df_result['Predicted_Phase'] = phase_results['Predicted_Phase'].values
        elif 'Phase' in phase_results.columns:
            df_result['Predicted_Phase'] = phase_results['Phase'].values
        else:
            for cand in ['predicted_phase', 'pred_phase', 'phase_pred']:
                if cand in phase_results.columns:
                    df_result['Predicted_Phase'] = phase_results[cand].values
                    break

        for conf_name in ['Phase_Confidence', 'Prediction_Confidence', 'pred_confidence', 'confidence']:
            if conf_name in phase_results.columns:
                df_result['Phase_Confidence'] = phase_results[conf_name].values
                break

        # ---------------------
        # Step 6: Stage-4 Dmax
        # ---------------------
        print("Step 6: Predicting Dmax (Stage-4)...")
        df_for_stage4 = df_feats.copy()
        df_for_stage4['Tg'] = pred_tg
        df_for_stage4['Tx'] = pred_tx
        df_for_stage4['Tl'] = pred_tl
        df_for_stage4['delta_Tx'] = delta_tx
        df_for_stage4['Trg'] = trg
        df_for_stage4['gamma'] = gamma

        try:
            dmax_results = s4_mod.predict_new_data(df_for_stage4)
        except TypeError:
            tmp4 = os.path.join(self.temp_dir, "step3_for_stage4.csv")
            df_for_stage4.to_csv(tmp4, index=False)
            dmax_results = s4_mod.predict_new_data(tmp4)

        if 'Predicted_Dmax' in dmax_results.columns:
            df_result['Predicted_Dmax_mm'] = dmax_results['Predicted_Dmax'].values
        elif 'Predicted_Dmax_mm' in dmax_results.columns:
            df_result['Predicted_Dmax_mm'] = dmax_results['Predicted_Dmax_mm'].values

        # ---------------------
        # Step 7: Stage-5 Rc
        # ---------------------
        print("Step 7: Predicting Rc (Stage-5)...")
        df_for_stage5 = df_for_stage4.copy()
        tmp5 = os.path.join(self.temp_dir, "step3_for_stage5.csv")
        df_for_stage5.to_csv(tmp5, index=False)
        rc_results = s5_mod.predict_new_data(tmp5)

        if 'Predicted_Rc' in rc_results.columns:
            df_result['Predicted_Rc_Ks'] = rc_results['Predicted_Rc'].values
        elif 'Predicted_Rc_Ks' in rc_results.columns:
            df_result['Predicted_Rc_Ks'] = rc_results['Predicted_Rc_Ks'].values

        df_result['Alloys'] = df_result['Alloys'].astype(str)

        # FINAL export or return
        if output_csv:
            df_result.to_csv(output_csv, index=False)
            print(f"\nSUCCESS: Final results saved to {output_csv}")
        else:
            print("\nSUCCESS: Final results prepared (not written to disk). Returning DataFrame.")
            self._cleanup()
            return df_result

        self._cleanup()
        return df_result

    def _cleanup(self):
        try:
            if os.path.isdir(self.temp_dir):
                for name in os.listdir(self.temp_dir):
                    path = os.path.join(self.temp_dir, name)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    os.rmdir(self.temp_dir)
                except Exception:
                    pass
        except Exception:
            pass


# Convenience wrapper
def predict_single_alloy(alloy_formula: str, out_csv: Optional[str] = None):
    pipeline = ModularBMGPipeline()
    return pipeline.run_pipeline(alloy_formula, output_csv=out_csv)
