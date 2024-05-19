import pandas as pd
from sklearn.model_selection import train_test_split

import my_constants as const


def split_train_test(data, test_size=const.TEST_SIZE, random_state=const.RANDOM_STATE):
    """
    Split the given data into training and testing sets.

    Parameters:
    - data: The input data to be split.
    - test_size: The proportion of the data to be used for testing. Default is const.TEST_SIZE.
    - random_state: The random seed for reproducibility. Default is const.RANDOM_STATE.

    Returns:
    - train: The training set.
    - test: The testing set.
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, test


def main():
    """
    This function reads the 'risklevel.csv' file, splits the data into train and test sets,
    and saves the train and test sets as separate CSV files.
    """

    processed_path = const.PROCESSED_PATH

    data = pd.read_csv(processed_path + "risklevel.csv")
    train, test = split_train_test(data)
    train.to_csv(processed_path + "risklevel_train.csv", index=False)
    test.to_csv(processed_path + "risklevel_test.csv", index=False)


if __name__ == "__main__":
    main()
