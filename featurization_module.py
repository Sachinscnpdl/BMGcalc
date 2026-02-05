# # Authored by: Sachin Poudel, Silesian University, Poland
"""
Stage-1 Featurization Module for BMG dataset
--------------------------------------------
Can be imported to featurize:
1. CSV files with "Alloys" column
2. Individual alloy composition strings
3. DataFrames with alloy compositions

Usage:
    from featurization_module import featurize_alloys, featurize_single_alloy
    
    # Option 1: Featurize a DataFrame
    df = pd.read_csv("alloys.csv")
    featurized_df = featurize_alloys(df, formula_col="Alloys")
    
    # Option 2: Featurize a single alloy string
    features = featurize_single_alloy("Zr65Cu15Ni10Al10")
    
    # Option 3: Featurize and save directly
    featurize_and_save("input.csv", "output.csv", formula_col="Alloys")
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Union, Tuple

# -----------------------
# Optional scientific stack
# -----------------------
try:
    from pymatgen.core import Composition
    from pymatgen.core.periodic_table import Element
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import (
        ElementProperty as _EP_base,
        Stoichiometry as _ST_base,
        ValenceOrbital as _VO_base,
        IonProperty as _IP_base,
    )
    MATMINER_AVAILABLE = True
except Exception:
    Composition = None
    Element = None
    StrToComposition = None
    _EP_base = _ST_base = _VO_base = _IP_base = None
    MATMINER_AVAILABLE = False

# -----------------------
# CONFIG
# -----------------------
R_GAS = 8.31446261815324
EPS = 1e-12

# Initialize Miedema table
MIEDEMA_TABLE = None

# Custom feature names (original 12 + added BMG-specific)
CUSTOM_COLS = [
    # Original core features (keep order)
    "mixing_entropy_J_per_molK",
    "VEC",
    "num_elements",
    "max_elem_fraction",
    "min_elem_fraction",
    "weighted_mean_atomic_number",
    "weighted_std_atomic_number",
    "weighted_mean_atomic_weight",
    "weighted_std_atomic_weight",
    "weighted_mean_electronegativity",
    "weighted_std_electronegativity",
    "atomic_size_mismatch_delta",
    # Added BMG-specific features
    "weighted_mean_atomic_radius",
    "weighted_std_atomic_radius",
    "weighted_range_atomic_radius",
    "weighted_mean_covalent_radius",
    "weighted_std_covalent_radius",
    "electronegativity_range",
    "pairwise_weighted_absdiff_chi",
    "Hmix_est_chi_sq",
    "Hmix_miedema",
    "Hmix_final",
    "Omega_parameter",
    "weighted_mean_melting_point",
    "weighted_std_melting_point",
    "fraction_transition_metal",
    "fraction_lanthanoid_actinoid",
    "fraction_metal",
    "atomic_mass_range",
]

# ============================================================
# INITIALIZATION FUNCTIONS
# ============================================================
def initialize_miedema_table(miedema_file: str = "miedema_params.csv") -> None:
    """
    Initialize the Miedema enthalpy parameters table.
    
    Parameters:
    -----------
    miedema_file : str, optional
        Path to CSV file with Miedema parameters (A, B, H columns)
    """
    global MIEDEMA_TABLE
    
    if os.path.exists(miedema_file):
        try:
            mdf = pd.read_csv(miedema_file)
            tbl = {}
            for _, r in mdf.iterrows():
                a = str(r["A"]).strip()
                b = str(r["B"]).strip()
                h = float(r["H"])
                tbl[frozenset({a, b})] = h
            MIEDEMA_TABLE = tbl
            print(f"Loaded Miedema table with {len(tbl)} pairs from {miedema_file}")
        except Exception as e:
            print(f"Failed to load {miedema_file}: {e}")
            MIEDEMA_TABLE = None
    else:
        MIEDEMA_TABLE = None
        print(f"Miedema file not found: {miedema_file}. Continuing without Miedema parameters.")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def print_nan_report(df: pd.DataFrame, title: str) -> None:
    """Print NaN statistics for a DataFrame."""
    print(f"\n--- NaN Report: {title} ---")
    total = len(df)
    for c in df.columns:
        nn = df[c].notna().sum()
        pct = 100 * nn / max(1, total)
        print(f"{c:35s} : {nn:6d}/{total}  ({pct:5.1f}%)")
    print("-" * 60)

def atomic_fraction_dict_from_composition(comp: Composition) -> Dict[str, float]:
    """Convert pymatgen Composition to atomic fraction dictionary."""
    if comp is None or Composition is None:
        return {}
    try:
        amt = comp.get_el_amt_dict()
        tot = sum(amt.values())
        if tot == 0:
            return {}
        return {k: v / tot for k, v in amt.items()}
    except Exception:
        return {}

def mixing_entropy_j_per_mol_k(frac: Dict[str, float]) -> float:
    """Calculate mixing entropy from atomic fractions."""
    vals = np.array([f for f in frac.values() if f > 0])
    if len(vals) == 0:
        return np.nan
    return float(-R_GAS * np.sum(vals * np.log(vals)))

def weighted_stat(frac: Dict[str, float], prop: Dict[str, float], 
                 stat: str = "mean") -> float:
    """Calculate weighted statistics for element properties."""
    fr, pr = [], []
    for el, f in frac.items():
        if el in prop and prop[el] is not None:
            fr.append(f)
            pr.append(prop[el])
    if not fr:
        return np.nan
    fr = np.array(fr)
    pr = np.array(pr, dtype=float)
    if stat == "mean":
        return float((fr * pr).sum())
    if stat == "std":
        m = (fr * pr).sum()
        return float(np.sqrt(((fr * (pr - m)) ** 2).sum()))
    if stat == "min":
        return float(np.min(pr))
    if stat == "max":
        return float(np.max(pr))
    if stat == "range":
        return float(np.max(pr) - np.min(pr))
    raise ValueError(f"Unknown stat: {stat}")

def _get_element_property_if_exists(el_symbol: str, attr_names: List[str], 
                                   fallback_attrs: Optional[List[str]] = None) -> Optional[float]:
    """Get element property from pymatgen Element object."""
    try:
        E = Element(el_symbol)
    except Exception:
        return None
    
    for a in attr_names:
        v = getattr(E, a, None)
        if v is not None:
            if isinstance(v, (list, tuple)):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    return float(v[0])
                continue
            try:
                return float(v)
            except Exception:
                return v
    
    if fallback_attrs:
        for a in fallback_attrs:
            v = getattr(E, a, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                        return float(v[0])
                    continue
    return None

def compute_valence_electron_count(comp: Composition) -> float:
    """Calculate valence electron count (VEC)."""
    if comp is None or Element is None:
        return np.nan
    frac = atomic_fraction_dict_from_composition(comp)
    vecs, fracs = [], []
    for el, f in frac.items():
        try:
            E = Element(el)
            fes = E.full_electronic_structure
            max_n = max(e[0] for e in fes if len(e) >= 3)
            val = sum(e[2] for e in fes if e[0] >= max_n - 1)
            vecs.append(val)
            fracs.append(f)
        except Exception:
            return np.nan
    return float(np.dot(vecs, fracs))

def atomic_size_mismatch_delta(comp: Composition) -> float:
    """Calculate atomic size mismatch parameter (δ)."""
    if comp is None or Element is None:
        return np.nan
    frac = atomic_fraction_dict_from_composition(comp)
    radii = {}
    for el in frac:
        try:
            E = Element(el)
            r = next(
                (getattr(E, a) for a in ("atomic_radius", "atomic_radius_calculated", 
                                        "metallic_radius", "covalent_radius")
                 if getattr(E, a, None) is not None),
                None,
            )
            if r is None:
                return np.nan
            radii[el] = float(r)
        except Exception:
            return np.nan
    r_vals = np.array(list(radii.values()))
    c_vals = np.array([frac[e] for e in radii])
    r_bar = (r_vals * c_vals).sum()
    return float(np.sqrt(((c_vals * (1 - r_vals / r_bar)) ** 2).sum()))

def compute_pairwise_weighted(frac: Dict[str, float], prop_dict: Dict[str, float], 
                             metric=lambda a, b: abs(a - b), scale_factor: float = 4.0) -> float:
    """Compute pairwise weighted property differences."""
    els = [el for el in frac if el in prop_dict and prop_dict[el] is not None]
    if len(els) < 2:
        return np.nan
    s = 0.0
    for i, j in combinations(els, 2):
        s += scale_factor * frac[i] * frac[j] * metric(prop_dict[i], prop_dict[j])
    return float(s)

def compute_miedema_Hmix(comp: Composition, miedema_table: Dict) -> Optional[float]:
    """Calculate mixing enthalpy using Miedema model."""
    if comp is None or miedema_table is None:
        return None
    frac = atomic_fraction_dict_from_composition(comp)
    H = 0.0
    any_missing = False
    for i, j in combinations(frac.keys(), 2):
        key = frozenset({i, j})
        if key in miedema_table:
            H += 4.0 * frac[i] * frac[j] * miedema_table[key]
        else:
            any_missing = True
    return float(H) if not any_missing else None

# ============================================================
# CORE FEATURIZATION FUNCTIONS
# ============================================================
def compute_all_custom_features(comp: Composition) -> pd.Series:
    """
    Compute all custom features for a given composition.
    
    Returns:
    --------
    pd.Series with index=CUSTOM_COLS
    """
    if comp is None or Element is None:
        return pd.Series([np.nan] * len(CUSTOM_COLS), index=CUSTOM_COLS)
    
    frac = atomic_fraction_dict_from_composition(comp)
    if not frac:
        return pd.Series([np.nan] * len(CUSTOM_COLS), index=CUSTOM_COLS)

    # --- base property dictionaries
    prop_z = {}
    prop_mass = {}
    prop_chi = {}
    prop_atomic_radius = {}
    prop_covalent = {}
    prop_melting = {}

    for el in frac:
        try:
            E = Element(el)
            prop_z[el] = getattr(E, "Z", None)
        except Exception:
            prop_z[el] = None
        prop_mass[el] = _get_element_property_if_exists(el, ["atomic_mass", "mass"])
        prop_chi[el] = _get_element_property_if_exists(el, ["X", "electronegativity", "pauling"])
        prop_atomic_radius[el] = _get_element_property_if_exists(
            el, ["atomic_radius", "atomic_radius_calculated", "metallic_radius"], 
            fallback_attrs=["covalent_radius"]
        )
        prop_covalent[el] = _get_element_property_if_exists(
            el, ["covalent_radius"], 
            fallback_attrs=["atomic_radius", "metallic_radius"]
        )
        prop_melting[el] = _get_element_property_if_exists(el, ["melting_point", "melt"])

    # --- original core features
    mixS = mixing_entropy_j_per_mol_k(frac)
    vec = compute_valence_electron_count(comp)
    n_elem = len(frac)
    maxf = max(frac.values())
    minf = min(frac.values())

    w_mean_Z = weighted_stat(frac, prop_z, "mean")
    w_std_Z = weighted_stat(frac, prop_z, "std")
    w_mean_mass = weighted_stat(frac, prop_mass, "mean")
    w_std_mass = weighted_stat(frac, prop_mass, "std")
    w_mean_chi = weighted_stat(frac, prop_chi, "mean")
    w_std_chi = weighted_stat(frac, prop_chi, "std")
    delta = atomic_size_mismatch_delta(comp)

    # --- added BMG-focused features
    w_mean_r = weighted_stat(frac, prop_atomic_radius, "mean")
    w_std_r = weighted_stat(frac, prop_atomic_radius, "std")
    w_range_r = weighted_stat(frac, prop_atomic_radius, "range")
    w_mean_covr = weighted_stat(frac, prop_covalent, "mean")
    w_std_covr = weighted_stat(frac, prop_covalent, "std")

    en_vals = [v for v in prop_chi.values() if v is not None]
    en_range = float(max(en_vals) - min(en_vals)) if len(en_vals) >= 2 else np.nan

    # pairwise descriptors
    pw_absdiff_chi = compute_pairwise_weighted(frac, prop_chi, metric=lambda a, b: abs(a - b))
    
    # heuristic Hmix using (Δχ)^2
    def sq_metric(a, b): 
        return (a - b) ** 2
    Hmix_est = compute_pairwise_weighted(frac, prop_chi, metric=sq_metric)

    # Miedema physical Hmix (if table present)
    global MIEDEMA_TABLE
    Hmix_phys = compute_miedema_Hmix(comp, MIEDEMA_TABLE) if MIEDEMA_TABLE is not None else None
    Hmix_final = Hmix_phys if Hmix_phys is not None else Hmix_est

    Tm_mean = weighted_stat(frac, prop_melting, "mean")
    Tm_std = weighted_stat(frac, prop_melting, "std")

    Omega = np.nan
    if (Tm_mean is not None) and (not np.isnan(Tm_mean)) and (Hmix_final is not None) and (not np.isnan(Hmix_final)) and abs(Hmix_final) > EPS:
        Omega = float(Tm_mean * mixS / (abs(Hmix_final) + EPS))

    # element class fractions
    n_total = len(frac)
    n_trans = n_lan_act = n_metal = 0
    for el in frac:
        try:
            E = Element(el)
            if getattr(E, "is_transition_metal", False):
                n_trans += 1
            if getattr(E, "is_lanthanoid", False) or getattr(E, "is_actinoid", False) or getattr(E, "is_lanthanoid_or_actinoid", False):
                n_lan_act += 1
            if getattr(E, "is_metal", False):
                n_metal += 1
        except Exception:
            pass
    frac_trans = float(n_trans / n_total) if n_total > 0 else np.nan
    frac_lan_act = float(n_lan_act / n_total) if n_total > 0 else np.nan
    frac_metal = float(n_metal / n_total) if n_total > 0 else np.nan

    mass_vals = [v for v in prop_mass.values() if v is not None]
    mass_range = float(max(mass_vals) - min(mass_vals)) if len(mass_vals) >= 2 else np.nan

    # --- assemble the series in CUSTOM_COLS order
    vals = [
        mixS,
        vec,
        n_elem,
        maxf,
        minf,
        w_mean_Z,
        w_std_Z,
        w_mean_mass,
        w_std_mass,
        w_mean_chi,
        w_std_chi,
        delta,
        w_mean_r,
        w_std_r,
        w_range_r,
        w_mean_covr,
        w_std_covr,
        en_range,
        pw_absdiff_chi,
        Hmix_est,
        Hmix_phys,
        Hmix_final,
        Omega,
        Tm_mean,
        Tm_std,
        frac_trans,
        frac_lan_act,
        frac_metal,
        mass_range,
    ]

    return pd.Series(vals, index=CUSTOM_COLS)

def drop_constant_or_nan_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns that are constant or all NaN."""
    drop = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0 or s.nunique() == 1:
            drop.append(c)
    return df.drop(columns=drop), drop

# ============================================================
# MAIN FEATURIZATION FUNCTIONS
# ============================================================
def featurize_single_alloy(alloy_string: str) -> Dict[str, float]:
    """
    Featurize a single alloy composition string.
    
    Parameters:
    -----------
    alloy_string : str
        Alloy composition (e.g., "Zr65Cu15Ni10Al10")
    
    Returns:
    --------
    Dict[str, float]: Dictionary of feature names and values
    """
    # Create a temporary DataFrame with one row
    df = pd.DataFrame({"Alloys": [alloy_string]})
    
    # Featurize using the DataFrame function
    result = featurize_alloys(df, formula_col="Alloys", save_targets=False)
    
    if len(result) == 0:
        return {}
    
    # Convert first row to dictionary
    features = result.iloc[0].to_dict()
    
    # Remove the original alloy string
    if "Alloys" in features:
        del features["Alloys"]
    
    return features

def featurize_alloys(df: pd.DataFrame, 
                     formula_col: str = "Alloys",
                     save_targets: bool = False,
                     targets_output: str = "targets_full.csv",
                     add_thermal_features: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Main function to featurize a DataFrame containing alloy compositions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing alloy compositions
    formula_col : str
        Column name containing alloy strings (default: "Alloys")
    save_targets : bool
        Whether to save target columns to separate CSV
    targets_output : str
        Output path for target CSV (if save_targets=True)
    add_thermal_features : bool
        Whether to compute thermal features (gamma, Trg, delta_Tx)
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    pd.DataFrame: Featurized DataFrame
    """
    if verbose:
        print("\n========== STAGE-1 FEATURIZATION ==========")
        print(f"Input DataFrame shape: {df.shape}")
    
    # Check if required column exists
    if formula_col not in df.columns:
        raise ValueError(f"Column '{formula_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # -----------------------
    # Save original targets if requested
    # -----------------------
    if save_targets:
        target_cols = [c for c in ["Tg", "Tx", "Tl", "Rc", "Dmax", "Phase"] if c in df.columns]
        if target_cols:
            df[target_cols].to_csv(targets_output, index=False)
            if verbose:
                print(f"Saved raw targets → {targets_output}")
    
    # -----------------------
    # Thermal parameters
    # -----------------------
    if add_thermal_features:
        for c in ["Tg", "Tx", "Tl", "Rc", "Dmax"]:
            if c in result_df.columns:
                result_df[c] = pd.to_numeric(result_df[c], errors="coerce")
        
        if {"Tg", "Tx", "Tl"}.issubset(result_df.columns):
            result_df["gamma"] = result_df["Tx"] / (result_df["Tg"] + result_df["Tl"])
            result_df["Trg"] = result_df["Tg"] / result_df["Tl"]
            result_df["delta_Tx"] = result_df["Tx"] - result_df["Tg"]
            
            if verbose:
                print_nan_report(result_df[["Tg", "Tx", "Tl", "gamma", "Trg", "delta_Tx"]], 
                               "Thermal features")
    
    # -----------------------
    # Composition parsing
    # -----------------------
    if StrToComposition is not None and MATMINER_AVAILABLE:
        if verbose:
            print("Parsing alloy compositions using matminer...")
        result_df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
            result_df, formula_col, ignore_errors=True
        )
    else:
        result_df["composition_obj"] = None
        if verbose:
            print("Matminer not available. Using basic composition parsing.")
    
    # -----------------------
    # Matminer featurizers
    # -----------------------
    if MATMINER_AVAILABLE:
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
        
        if verbose:
            print(f"Matminer features added. Total columns now: {result_df.shape[1]}")
    
    # -----------------------
    # Custom features
    # -----------------------
    if verbose:
        print("Computing custom physics features...")
    
    custom = result_df["composition_obj"].apply(compute_all_custom_features)
    result_df = pd.concat([result_df, custom], axis=1)
    
    if verbose:
        print_nan_report(result_df[CUSTOM_COLS], "Custom physics features")
    
    # -----------------------
    # Cleanup
    # -----------------------
    result_df.drop(columns=["composition_obj", formula_col], inplace=True, errors="ignore")
    result_df, dropped = drop_constant_or_nan_columns(result_df)
    
    if verbose:
        print(f"\nDropped {len(dropped)} constant/NaN columns")
        print(f"Final featurized shape: {result_df.shape}")
        print("========== FEATURIZATION COMPLETE ==========\n")
    
    return result_df

def featurize_and_save(input_file: str, 
                      output_file: str, 
                      formula_col: str = "Alloys",
                      targets_output: Optional[str] = None,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Load, featurize, and save alloy data in one step.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to save featurized CSV
    formula_col : str
        Column name containing alloy strings
    targets_output : str, optional
        Path to save target columns (if None, won't save)
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    pd.DataFrame: Featurized DataFrame
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Featurize
    featurized = featurize_alloys(
        df, 
        formula_col=formula_col,
        save_targets=targets_output is not None,
        targets_output=targets_output or "targets_full.csv",
        verbose=verbose
    )
    
    # Save
    featurized.to_csv(output_file, index=False)
    
    if verbose:
        print(f"Featurized data saved to: {output_file}")
    
    return featurized

# ============================================================
# INITIALIZE ON IMPORT
# ============================================================
# Initialize Miedema table when module is imported
initialize_miedema_table()

# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Example 1: Featurize a CSV file
    print("Example 1: Featurizing from CSV file")
    try:
        example_df = pd.DataFrame({
            "Alloys": ["Zr65Cu15Ni10Al10", "Cu50Zr50", "Fe80B20", "Mg65Cu25Y10"],
            "Tg": [650, 700, 750, 600],
            "Tx": [700, 750, 800, 650],
            "Tl": [1100, 1200, 1500, 900]
        })
        
        featurized = featurize_alloys(example_df, verbose=True)
        print("\nFirst few rows of featurized data:")
        print(featurized.head())
        
        # Save to file
        featurized.to_csv("example_featurized.csv", index=False)
        print("\nSaved example to 'example_featurized.csv'")
        
    except Exception as e:
        print(f"Error in example: {e}")
    
    # Example 2: Featurize single alloy
    print("\n\nExample 2: Featurizing single alloy")
    features = featurize_single_alloy("Zr65Cu15Ni10Al10")
    print("\nFeatures for Zr65Cu15Ni10Al10:")
    for key, value in list(features.items())[:10]:  # Show first 10
        print(f"  {key}: {value}")
    
    print("\n... and", len(features) - 10, "more features")
