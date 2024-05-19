# RandomForest

## Table of Contents

- [RandomForest](#randomforest)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Directory Structure](#directory-structure)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contribution](#contribution)

## About <a name = "about"></a>

Detect suicidal risk in hotline calls using a Random Forest model. It includes data preprocessing, model training, evaluation, and analysis.

## Directory Structure

/RandomForest
├── data
│   ├── rawdata
│   └── processed
├── models
├── notebooks
│   └── analysis/*.r
└── src
    ├── data_processing.py
    ├── split_train_test.py
    ├── models_rf.py
    ├── my_constants.py
    └── my_utils.py

## Features

- **Data Preprocessing**: `data_processing.py` & `split_train_test.py`
- **Model Training and Evaluation**: `models_rf.py`
- **Analysis and Visualization**: `notebooks/analysis/*.r`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/110752002/LifelineSuicidePrediction.git
   ```
2. Navigate to the RandomForest directory:
   ```bash
   cd LifelineSuicidePrediction/RandomForest
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Set Hyperparameter in `src/my_constants.py`**
2. **Preprocess Data**:
  ```bash
  python src/data_processing.py
  ```
3. **Split Training & Testing Data**
  ```bash
  python src/split_train_test.py
  ```
4. **Train & Evaluate Model**:
  ```bash
  python src/train_model.py
  ```

## Contribution

Feel free to fork this repository and contribute via pull requests. For major changes, please open an issue first to discuss what you would like to change.
