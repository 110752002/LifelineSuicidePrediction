import pandas as pd
import os

from sklearn import pipeline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import ensemble
from sklearn import model_selection

import time
import joblib
import tqdm

from imblearn.over_sampling import RandomOverSampler

import my_constants as const
import my_utils as utils


# Select the top k utterances with the highest similarity scores of a specific reference
def Select_key_utterances(df, which_ref, top_k=10):
    """
    Selects the top k key utterances from the given DataFrame based on the specified reference column.

    Args:
        - df (pandas.DataFrame): The DataFrame containing the utterances.
        - which_ref (str): The name of the reference column to sort the DataFrame by.
        Options: ["max_sim_ref1", "avg_sim_ref1", "max_sim_ref2", "avg_sim_ref2", "max_sim_ref3", "avg_sim_ref3", "max_sim_ref4", "avg_sim_ref4", ...]
        - top_k (int, optional): The number of key utterances to select. Defaults to 10.

    Returns:
        - pandas.DataFrame: A new DataFrame containing the top k key utterances sorted by the specified reference column.
            Dimension: (top_k, df.columns).
    """
    return (
        df.sort_values(by=which_ref, ascending=False).head(top_k).reset_index(drop=True)
    )


# Extracting embeddings of the top k key utterances as features
def Extract_key_utterance_emb(df, which_ref, emb_col="emb", top_k=10):
    """
    Extracts the key utterance embeddings from a DataFrame.

    Args:
      - df (pandas.DataFrame): The input DataFrame containing the utterance data.
      - which_ref (str): The reference for selecting the key utterances.
          Options: ["max_sim_ref1", "avg_sim_ref1", "max_sim_ref2", "avg_sim_ref2", "max_sim_ref3", "avg_sim_ref3", "max_sim_ref4", "avg_sim_ref4", ...]
      - emb_col (str, optional): The name of the column containing the embeddings. Defaults to "emb".
      - top_k (int, optional): The number of top key utterances to select. Defaults to 10.

    Returns:
      - pandas.DataFrame: The DataFrame with the extracted key utterance embeddings.
          Dimension: (df.dialogue_id.nunique(), emb_col*top_k).
    """
    # Select the top k key utterances each transcript
    df = Select_key_utterances(df, which_ref, top_k)
    # Concatenate the embeddings of the top k key utterances of one string each transcript
    df = (
        df.groupby("dialogue_id")[emb_col]
        .apply(
            lambda x: " ".join(x),
        )
        .reset_index()
    )
    # Expand each dimension of the concatenated embeddings into a separate column
    df = (
        df[emb_col]
        .apply(lambda x: pd.Series(utils.Convert_str_to_list(x)))
        .join(df["dialogue_id"])
    )

    return df


# Resamples the data to deal with class imbalance using RandomOverSampler.
def Deal_with_imbalance(x, y):
    """
    Resamples the data to deal with class imbalance using RandomOverSampler.

    Args:
        - x (array-like): The input features.
        - y (array-like): The target variable.

    Returns:
        - x_resampled (array-like): The resampled input features.
        - y_resampled (array-like): The resampled target variable.
    """
    # Create a RandomOverSampler object
    ros = RandomOverSampler(random_state=const.RANDOM_STATE)

    # Fit the RandomOverSampler object to the data
    x_resampled, y_resampled = ros.fit_resample(x, y)

    return x_resampled, y_resampled


# Create a pipeline for standarizing, pca, and classifying
def Create_pipeline():
    """
    Create a pipeline for preprocessing and classification using Random Forest.

    Returns:
        pipeline (Pipeline): A scikit-learn pipeline object.
    """
    # Create a pipeline
    pipeline = pipeline.Pipeline(
        [
            ("scaler", preprocessing.StandardScaler()),
            (
                "pca",
                decomposition.PCA(svd_solver="full", random_state=const.RANDOM_STATE),
            ),
            (
                "clf",
                ensemble.RandomForestClassifier(
                    n_estimators=const.N_ESTIMATORS, random_state=const.RANDOM_STATE
                ),
            ),
        ]
    )

    return pipeline


# Create a grid search object
def Create_grid_search(pipeline):
    """
    Create a grid search object.

    Args:
        - pipeline (object): The pipeline object to be used for grid search.

    Returns:
        - grid_search (object): The grid search object.

    """
    # Create a grid search object
    grid_search = model_selection.GridSearchCV(
        pipeline, const.PARAM_GRID, cv=const.CV_FOlDS, scoring=const.CV_SCORE, n_jobs=-1
    )

    return grid_search


# Select cum_ut_i utterances from the beginning of the transcript
def Select_cum_utterances(df, cum_ut_i):
    """
    Selects the first cum_ut_i utterances from the given DataFrame.

    Args:
        - df (pandas.DataFrame): The DataFrame containing the utterances.
        - cum_ut_i (int): The number of utterances to select.

    Returns:
        - pandas.DataFrame: A new DataFrame containing the first cum_ut_i utterances.
            Dimension: (cum_ut_i*transcript_num, column).
    """
    return df.groupby("dialogue_id").head(cum_ut_i).reset_index(drop=True)


def main():
    """
    Main function for training and evaluating a random forest model for suicide prediction.

    This function performs the following steps:
    1. Loads the preprocessed data.
    2. Extracts key utterance embeddings as features.
    3. Trains a random forest model using grid search and saves the best model.
    4. Predicts the test data using the best model and calculates evaluation metrics.
    5. Saves the evaluation metrics and predictions for cumulatively input utterances.

    Returns:
        None
    """
    total_time0 = time.time()

    processed_path = const.PROCESSED_PATH
    model_path = const.MODEL_PATH
    report_path = const.REPORT_PATH

    # Load the preprocessed data

    transcript_df = pd.read_csv(processed_path + "transcripts_emb_caller.csv")
    risklevel_train_df = pd.read_csv(processed_path + "risklevel_train.csv")
    risklevel_test_df = pd.read_csv(processed_path + "risklevel_test.csv")

    # Maximum number of utterances
    max_ut_len = transcript_df.groupby("dialogue_id").size().max()
    print(f"Maximum number of utterances: {max_ut_len}\n")

    # for loop for each reference column and risk level column
    for ref_col_i in const.REF_COL_LI:
        for risk_col_i in const.RISK_LEVEL_COL_LI:

            print(f"Reference: {ref_col_i}, Risk level: {risk_col_i}\n")

            # Extract the embeddings of the top k key utterances as features

            expandedTranscript_df = (
                transcript_df.groupby("dialogue_id")
                .apply(
                    lambda x: Extract_key_utterance_emb(x, ref_col_i, top_k=const.TOP_K)
                )
                .reset_index(drop=True)
            )

            # Merge the expanded transcript with the risk level data (dialogue_id as the key)
            expandedTranscript_train_df = risklevel_train_df[
                ["dialogue_id", risk_col_i]
            ].merge(expandedTranscript_df, on="dialogue_id")
            expandedTranscript_test_df = risklevel_test_df[
                ["dialogue_id", risk_col_i]
            ].merge(expandedTranscript_df, on="dialogue_id")

            x_train = expandedTranscript_train_df.drop(
                ["dialogue_id", risk_col_i], axis=1
            )
            y_train = expandedTranscript_train_df[risk_col_i]
            x_test = expandedTranscript_test_df.drop(
                ["dialogue_id", risk_col_i], axis=1
            )
            y_test = expandedTranscript_test_df[risk_col_i]

            # Resample the data to deal with class imbalance
            x_train_resampled, y_train_resampled = Deal_with_imbalance(x_train, y_train)

            #### Random Forest Traning ########################################################
            # Create a pipeline
            pipeline_obj = Create_pipeline()

            # Create a grid search object
            grid_search_obj = Create_grid_search(pipeline_obj)

            time0 = time.time()
            print(f"Start training...\n")

            # Fit the grid search object to the data
            grid_search_obj.fit(x_train_resampled, y_train_resampled)

            time1 = time.time()
            print(f"Training time elapsed: {utils.Format_time(time1 - time0)}\n")

            # Get the best parameters and the best model
            best_params = grid_search_obj.best_params_
            best_model = grid_search_obj.best_estimator_

            # Save the best model
            model_name = f"rf_{risk_col_i}_{ref_col_i}.joblib"
            joblib.dump(best_model, model_path + model_name)

            # Predict the test data
            y_pred = best_model.predict(x_test)

            # Calculate all the metrics
            accuracy, precision, recall, f1, grade_precision, grade_recall, FScore = (
                utils.CalcAllScores(y_test, y_pred, avg=const.AVG_METHOD)
            )
            print(
                f"Predict the test data of reference: {ref_col_i}, risk level: {risk_col_i}\n",
                f"Best parameters: {best_params}\n",
                f"Accuracy: {accuracy:.4f}\n",
                f"Precision: {precision:.4f}\n",
                f"Recall: {recall:.4f}\n",
                f"F1: {f1:.4f}\n",
                f"Graded precision: {grade_precision:.4f}\n",
                f"Graded recall: {grade_recall:.4f}\n",
                f"FScore: {FScore:.4f}\n",
            )

            # Save the report
            report_name = f"full_rf_{risk_col_i}_{ref_col_i}.csv"

            report_df = pd.DataFrame(
                {
                    "reference": [ref_col_i],
                    "risk_col": [risk_col_i],
                    "best_params": [best_params],
                    "accuracy": [accuracy],
                    "precision": [precision],
                    "recall": [recall],
                    "f1": [f1],
                    "grade_precision": [grade_precision],
                    "grade_recall": [grade_recall],
                    "FScore": [FScore],
                }
            )
            report_df.to_csv(report_path + report_name, index=False)

            #### Predict the test data by cumulatively input utterances ########################

            time0 = time.time()
            print(
                f"Start predicting the test data by cumulatively input utterances...\n"
            )

            # Create a list to store the predictions
            predict_label_df = risklevel_test_df[["dialogue_id", risk_col_i]]
            predict_metric_df = pd.DataFrame(
                columns=[
                    "cum_ut_i",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "grade_precision",
                    "grade_recall",
                    "FScore",
                ]
            )

            for cum_ut_i in tqdm.tqdm(
                range(10, max_ut_len + const.CUM_STEP, const.CUM_STEP)
            ):

                # Select cum_ut_i utterances from the beginning of the transcript
                cum_transcript_df = Select_cum_utterances(transcript_df, cum_ut_i)

                # Extract the embeddings of the top k key utterances as features
                cum_expandedTranscript_test_df = risklevel_test_df[
                    ["dialogue_id", risk_col_i]
                ].merge(
                    cum_transcript_df.groupby("dialogue_id")
                    .apply(
                        lambda x: Extract_key_utterance_emb(
                            x, ref_col_i, top_k=const.TOP_K
                        )
                    )
                    .reset_index(drop=True),
                    on="dialogue_id",
                )

                cum_x_test = cum_expandedTranscript_test_df.drop(
                    ["dialogue_id", risk_col_i], axis=1
                )
                cum_y_test = cum_expandedTranscript_test_df[risk_col_i]

                # Predict the test data
                cum_y_pred = best_model.predict(cum_x_test)

                # Calculate all the metrics
                (
                    accuracy,
                    precision,
                    recall,
                    f1,
                    grade_precision,
                    grade_recall,
                    FScore,
                ) = utils.CalcAllScores(cum_y_test, cum_y_pred, avg=const.AVG_METHOD)

                # Save the metrics
                predict_label_df[f"{cum_ut_i}"] = cum_y_pred

                # 假如cum_ut_i = 10 就建立一個新的 DataFrame
                if cum_ut_i == 10:
                    predict_metric_df = pd.DataFrame(
                        {
                            "cum_ut_i": [cum_ut_i],
                            "accuracy": [accuracy],
                            "precision": [precision],
                            "recall": [recall],
                            "f1": [f1],
                            "grade_precision": [grade_precision],
                            "grade_recall": [grade_recall],
                            "FScore": [FScore],
                        }
                    )
                else:
                    new_metric_df = pd.DataFrame(
                        {
                            "cum_ut_i": [cum_ut_i],
                            "accuracy": [accuracy],
                            "precision": [precision],
                            "recall": [recall],
                            "f1": [f1],
                            "grade_precision": [grade_precision],
                            "grade_recall": [grade_recall],
                            "FScore": [FScore],
                        }
                    )
                    predict_metric_df = pd.concat(
                        [predict_metric_df, new_metric_df], ignore_index=True
                    )

            time1 = time.time()
            print(
                f"Prediction by cumulatively input utterances time elapsed: {utils.Format_time(time1 - time0)}\n"
            )

            # Save the cumulatively input utterances predictions and the metrics
            predict_label_df = predict_label_df.melt(
                id_vars=["dialogue_id", risk_col_i],
                var_name="cum_ut",
                value_name="y_pred",
            )  # to long format
            predict_label_name = f"cum_rf_{risk_col_i}_{ref_col_i}_predict_label.csv"
            predict_label_df.to_csv(report_path + predict_label_name, index=False)
            predict_metric_name = f"cum_rf_{risk_col_i}_{ref_col_i}_predict_metric.csv"
            predict_metric_df.to_csv(report_path + predict_metric_name, index=False)

    total_time1 = time.time()
    print(f"Total time elapsed: {utils.Format_time(total_time1 - total_time0)}\n")


if __name__ == "__main__":
    main()
    print("Random Forest model training and prediction are done.")
