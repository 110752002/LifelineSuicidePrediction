import datetime
from sklearn import metrics


import datetime

def Format_time(elapsed):
    """
    Takes a time in seconds and returns a string in the format hh:mm:ss.

    Args:
    elapsed (int): The time in seconds.

    Returns:
    str: The formatted time string in the format hh:mm:ss.
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Helper fuction for converting embedding from a string to a list of floats
def Convert_str_to_list(emb_str):
    """
    Convert a string representation of a embedding list of numbers to a list of floats.

    Args:
        emb_str (str): The string representation of the embedding list.

    Returns:
        list: A list of floats.

    Example:
        >>> emb_str = "[1.0, 2.0, 3.0]"
        >>> Convert_str_to_list(emb_str)
        [1.0, 2.0, 3.0]
    """
    emb_str = emb_str.replace("[", "").replace("]", "").replace("\n", " ")
    return [float(item) for item in emb_str.split() if item]


def CalcGradedPrecision(y_true, y_pred):
    """
    Calculate the graded precision of a multi-class classification model. See Tsai & Chen (2022) for more details.
    
    Args:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.

    Returns:
    - graded_precision (float): The graded precision score.

    """
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    new_fn = sum(
        true_one > pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )  # false negative: underestimation of risk
    new_fp = sum(
        true_one < pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )  # false positive: overestimation of risk
    new_tp = sum(
        true_one == pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )

    graded_precision = new_tp / (new_tp + new_fp)

    return graded_precision


def CalcGradedRecall(y_true, y_pred):
    """
    Calculate the graded recall metric for a multi-class classification model. See Tsai & Chen (2022) for more details.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.

    Returns:
        float: The graded recall score.

    """
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    new_fn = sum(
        true_one > pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )  # false negative: underestimation of risk
    new_fp = sum(
        true_one < pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )  # false positive: overestimation of risk
    new_tp = sum(
        true_one == pred_one for true_one, pred_one in zip(y_true_int, y_pred_int)
    )

    graded_recall = new_tp / (new_tp + new_fn)

    return graded_recall


def CalcFScore(y_true, y_pred):
    """
    Calculate the F-Score given the true labels and predicted labels. See Tsai & Chen (2022) for more details.

    Args:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    Returns:
    - FScore: The F-Score calculated using the graded precision and graded recall.
    """

    FScore = (
        2
        * (CalcGradedPrecision(y_true, y_pred) * CalcGradedRecall(y_true, y_pred))
        / (CalcGradedPrecision(y_true, y_pred) + CalcGradedRecall(y_true, y_pred))
    )

    return FScore


# Helper function for calculating all the score (accuracy, precision, recall, F1, grade_recall, grade_precision, FScore)
def CalcAllScores(y_true, y_pred, avg="macro"):
    """
    Calculate various evaluation scores for a classification model.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - avg (str, optional): The averaging strategy for multiclass classification.
                           Default is "macro".

    Returns:
    - accuracy (float): Accuracy score.
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1 (float): F1 score.
    - grade_precision (float): Graded precision score.
    - grade_recall (float): Graded recall score.
    - FScore (float): F-Score.

    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=avg)
    recall = metrics.recall_score(y_true, y_pred, average=avg)
    f1 = metrics.f1_score(y_true, y_pred, average=avg)
    grade_precision = CalcGradedPrecision(y_true, y_pred)
    grade_recall = CalcGradedRecall(y_true, y_pred)
    FScore = CalcFScore(y_true, y_pred)

    return accuracy, precision, recall, f1, grade_precision, grade_recall, FScore
