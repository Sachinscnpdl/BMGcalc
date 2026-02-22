import os
import trained_stage2_with_predictions_flexible as s2_train

# Check if the Stage 2 artifacts exist
if not os.path.exists("/mount/src/bmgcalc/STAGE2_classificatuion/artifacts.joblib"):
    with st.spinner("Initializing models for the first time... this may take a minute."):
        s2_train.train_model() # This runs your training script on the server
