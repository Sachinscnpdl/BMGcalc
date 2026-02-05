# Authored by: Sachin Poudel, Silesian University, Poland
import os
import tempfile
import numpy as np
import pandas as pd

def process_alloys(input_data, output_file=None):
    """
    Process alloys from either a single alloy string or a CSV file with 'Alloys' column
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        - Single alloy string (e.g., "Zr65Cu15Ni10Al10")
        - Path to CSV file with 'Alloys' column
        - DataFrame with 'Alloys' column
    output_file : str, optional
        Path to save results CSV file
    
    Returns:
    --------
    pd.DataFrame: DataFrame with all predictions
    """

    
    # Determine input type and load data
    if isinstance(input_data, str):
        if input_data.endswith('.csv'):
            # It's a CSV file path
            df = pd.read_csv(input_data)
            print(f"Loaded {len(df)} alloys from CSV file: {input_data}")
        else:
            # It's a single alloy string
            df = pd.DataFrame({"Alloys": [input_data]})
            print(f"Processing single alloy: {input_data}")
    elif isinstance(input_data, pd.DataFrame):
        # It's already a DataFrame
        df = input_data.copy()
        print(f"Processing {len(df)} alloys from DataFrame")
    else:
        raise ValueError("Input must be a string (alloy or CSV path) or a pandas DataFrame")
    
    # Check if 'Alloys' column exists
    if 'Alloys' not in df.columns:
        raise ValueError("Input data must have an 'Alloys' column")
    
    # Create a temporary directory for our files
    temp_dir = tempfile.mkdtemp()
    temp_featurized_path = os.path.join(temp_dir, "featurized.csv")
    
    # Step 1: Featurize all alloys
    print("Step 1: Featurizing alloys...")
    featurized = featurize_alloys_complete(df, formula_col="Alloys")
    featurized.to_csv(temp_featurized_path, index=False)
    
    # Step 2: Predict thermal properties FIRST (needed for phase prediction)
    print("Step 2: Predicting thermal properties...")
    thermal_predictions = predict_thermal_for_alloys(temp_featurized_path)
    
    # Step 3: Add thermal predictions to the featurized data
    featurized_with_thermal = featurized.copy()
    featurized_with_thermal["Tg"] = thermal_predictions["Predicted_Tg"]
    featurized_with_thermal["Tx"] = thermal_predictions["Predicted_Tx"]
    featurized_with_thermal["Tl"] = thermal_predictions["Predicted_Tl"]
    
    # Save the updated featurized data with thermal properties
    featurized_with_thermal.to_csv(temp_featurized_path, index=False)
    
    # Step 4: Phase classification (now with thermal properties)
    print("Step 4: Classifying phases...")
    from trained_stage2_with_predictions_flexible import predict_new_data as predict_phase
    phase_predictions = predict_phase(temp_featurized_path)
    
    # Step 5: Critical diameter prediction
    print("Step 5: Predicting critical diameter...")
    from stage4_dmax import predict_new_data as predict_dmax
    dmax_predictions = predict_dmax(temp_featurized_path)
    
    # Step 6: Cooling rate prediction
    print("Step 6: Predicting cooling rate...")
    from stage5_rc import predict_new_data as predict_rc
    rc_predictions = predict_rc(temp_featurized_path)
    
    # Step 7: Combine all results
    print("Step 7: Combining results...")
    final_results = df.copy()
    
    # Add phase predictions
    final_results["Predicted_Phase"] = phase_predictions["Predicted_Phase"]
    final_results["Phase_Confidence"] = phase_predictions["Prediction_Confidence"]
    
    # Add thermal predictions
    final_results["Predicted_Tg"] = thermal_predictions["Predicted_Tg"]
    final_results["Predicted_Tx"] = thermal_predictions["Predicted_Tx"]
    final_results["Predicted_Tl"] = thermal_predictions["Predicted_Tl"]
    
    # Add Dmax predictions
    final_results["Predicted_Dmax"] = dmax_predictions["Predicted_Dmax"]
    
    # Add Rc predictions
    final_results["Predicted_Rc"] = rc_predictions["Predicted_Rc"]
    
    # Save results if output file is specified
    if output_file:
        final_results.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return final_results

def featurize_alloys_complete(df, formula_col="Alloys"):
    """Complete featurization for alloys that ensures all features match training data"""
    import numpy as np
    import pandas as pd
    
    # Load the original featurized data to see what features we need
    try:
        # Try to load the original featurized data
        original_df = pd.read_csv("featurized_metallic_glass_stage1.csv")
        required_features = list(original_df.columns)
    except:
        # If we can't load the original data, we'll proceed with what we have
        required_features = None
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if required column exists
    if formula_col not in result_df.columns:
        raise ValueError(f"Column '{formula_col}' not found in DataFrame")
    
    # Parse compositions
    from featurization_module import StrToComposition, MATMINER_AVAILABLE
    if MATMINER_AVAILABLE:
        result_df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
            result_df, formula_col, ignore_errors=True
        )
    else:
        result_df["composition_obj"] = None
    
    # Add matminer features
    if MATMINER_AVAILABLE:
        from featurization_module import (
            _EP_base, _ST_base, _VO_base, _IP_base
        )
        
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
    
    # Add custom features
    from featurization_module import compute_all_custom_features
    custom = result_df["composition_obj"].apply(compute_all_custom_features)
    result_df = pd.concat([result_df, custom], axis=1)
    
    # For single rows, skip the column dropping step
    if len(result_df) > 1:
        from featurization_module import drop_constant_or_nan_columns
        result_df, _ = drop_constant_or_nan_columns(result_df)
    
    # Drop the composition object column and original formula column
    result_df.drop(columns=["composition_obj", formula_col], inplace=True, errors="ignore")
    
    # If we have required features from the original data, ensure we have all of them
    if required_features is not None:
        # Add any missing features with NaN values
        for feature in required_features:
            if feature not in result_df.columns:
                result_df[feature] = np.nan
        
        # Reorder columns to match the original data
        # Only include columns that exist in both dataframes
        common_features = [f for f in required_features if f in result_df.columns]
        result_df = result_df[common_features]
    
    # Fill NaN values with 0 or median for numeric columns
    # This is crucial for the preprocessing pipeline
    for col in result_df.columns:
        if result_df[col].isna().any():
            # For numeric columns, fill with 0
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(0)
            else:
                # For categorical columns, fill with the most frequent value
                most_frequent = result_df[col].mode()
                if not most_frequent.empty:
                    result_df[col] = result_df[col].fillna(most_frequent[0])
                else:
                    result_df[col] = result_df[col].fillna("unknown")
    
    return result_df

def predict_thermal_for_alloys(featurized_path):
    """Modified thermal prediction function for multiple alloys"""
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    
    # Load the model artifacts
    from stage3_thermal_properties import STAGE1_MODEL, STAGE1_TOP, STAGE1_SCALER, GlassThermalModel, DEVICE
    
    # Load the featurized data
    df = pd.read_csv(featurized_path)
    
    # Load the trained model components
    top_features = np.load(STAGE1_TOP)
    scaler = joblib.load(STAGE1_SCALER)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    
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
    
    # Make predictions in batches if needed
    batch_size = 64
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(DEVICE)
            batch_pred = model(batch).cpu().numpy()
            predictions.append(batch_pred)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Create a DataFrame with predictions
    result_df = df.copy()
    result_df["Predicted_Tg"] = predictions[:, 0]
    result_df["Predicted_Tx"] = predictions[:, 1]
    result_df["Predicted_Tl"] = predictions[:, 2]
    
    return result_df
