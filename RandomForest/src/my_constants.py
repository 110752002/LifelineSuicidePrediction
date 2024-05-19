
"""
This module contains constant values used in Random Forest Model in the Lifeline Suicide Prediction project.
"""

RANDOM_STATE = 20240229

REF_COL_LI = list(
    [
        "max_sim_ref1", # Suicidal ideation scale + Suicide risk scale
        "avg_sim_ref1",
        "max_sim_ref2", # Suicidal ideation scale
        "avg_sim_ref2",
        "max_sim_ref3", # Suicide risk scale
        "avg_sim_ref3",
        "max_sim_ref4", # Lv et al, 2015. Suicide dict
        "avg_sim_ref4",
    ]
)
RISK_LEVEL_COL_LI = list(["risk3", "risk4", "risk5"])
TRANSCRIPT_EMB_CSV = "transcripts_emb_caller.csv"  # ["transcripts_emb_caller.csv", "transcripts_emb_all.csv", "transcripts_emb_responser.csv"]

# Path
PROCESSED_PATH = "data/processed/"
MODEL_PATH = "models/"
REPORT_PATH = "results/"


TOP_K = 10  # Select top `top_k` key utterances
CUM_STEP = 10  # The number of steps accumulated utterance by utterance


# Set random forest
N_ESTIMATORS = 500
TEST_SIZE = 0.2

# Set grid search
PARAM_GRID = {
    "pca__n_components": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # Components after Dimension reduction
    "clf__max_features": [1, 3, 5, 7, 9],  # Max features of each tree
}
CV_FOlDS = 5
CV_SCORE = "f1_macro"
AVG_METHOD = "macro"

