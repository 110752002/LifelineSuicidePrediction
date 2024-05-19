import pandas as pd
import glob
import torch
import sentence_transformers
import time

import my_utils as utils


def Read_transcripts(folder_path):
    """
    Read transcripts from multiple CSV files.

    Args:
      folder_path (str): The path to the directory containing the CSV files.

    Returns:
      dict: A dictionary where the keys are the filenames and the values are the corresponding pandas DataFrames.

    """
    # Read all transcript data
    x1 = glob.glob(folder_path + "transcript_UTF8_2_v1/*/*/*.csv")
    x2 = glob.glob(folder_path + "transcript_UTF8_2_v2/*/*/*.csv")
    x3 = glob.glob(folder_path + "transcript_UTF8_2_207/*/*/*.csv")

    data = {}
    for i in x1:
        data[i[-18:-4]] = pd.read_csv(i, encoding="utf-8")
    for i in x2:
        data[i[-18:-4]] = pd.read_csv(i, encoding="utf-8")
    for i in x3:
        data[i[-18:-4]] = pd.read_csv(i, encoding="utf-8")

    return data


def Read_risk_level(folder_path):
    """
    Read the suicidal risk level data from a CSV file.

    Args:
      folder_path (str): The path to the folder containing the CSV file.

    Returns:
      pandas.DataFrame: A DataFrame containing the risk level data with columns 'dialogue_id' and 'risk_level'.
    """
    data = pd.read_csv(folder_path + "risklevel.csv", header=None)
    data.columns = ["dialogue_id", "risk_level"]

    return data


def Clean_transcripts(data):
    """
    Clean transcripts.

    Args:
      data (dict): A dictionary where the keys are the filenames and the values are the corresponding pandas DataFrames.

    Returns:
      dict: A dictionary where the keys are the filenames and the values are the corresponding cleaned pandas DataFrames.

    """
    cleaned_data = {}
    for key, value in data.items():
        # Clean data
        cleaned_data[key] = Clean_one_transcript(data[key])

    return cleaned_data


def Clean_one_transcript(df):
    """
    Cleans and processes a transcript dataframe by removing rows with no dialogue, replacing tags and '#' with empty strings, and splitting the 'utterance' column into multiple rows.

    Args:
      df (pandas.DataFrame): The input dataframe containing the transcript.

    Returns:
      pandas.DataFrame: The cleaned and processed dataframe.

    """
    # Uniform column names
    df = df.iloc[:, [0, 1]]
    df.columns = ["role", "utterance"]  # Uniform column names
    df = df.dropna(axis=0)  # Remove rows with NaN values

    # Replace '<.+?>' and '#' strings in the dataframe with empty strings using regular expressions
    df.replace("<.+?>|#", "", regex=True, inplace=True)

    # Split the 'utterance' column using '.' and '?' as separators, and expand the result into multiple new columns
    step1 = df["utterance"].str.split("？|。", expand=True)

    # Reshape the DataFrame and convert it to a Series
    step2 = step1.stack()

    # Reset the index and remove the extra index level
    step3 = step2.reset_index(level=1, drop=True)
    step3.name = "utterance" # Rename the Series to 'utterance'

    # Remove the original 'utterance' column and join the new 'utterance' Series
    step4 = df.drop(["utterance"], axis=1).join(step3)

    # Reset the index of the entire DataFrame
    step5 = step4.reset_index()

    # Remove the 'index' column
    step6 = step5.drop("index", axis=1)

    # Remove rows with empty strings in the 'utterance' column
    step7 = step6.drop(step6[step6["utterance"] == ""].index)

    # Remove rows with NaN values
    step8 = step7.dropna(axis=0)

    return step8


def Clean_risk_level(data):
    """
    Cleans the input data by removing rows with risk levels other than 1-5.

    Args:
        data (pandas.DataFrame): The input data to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned data with only risk levels 1-5.
    """
    cleaned_data = data[data["risk_level"].isin(["1", "2", "3", "4", "5"])].reset_index(
        drop=True
    )
    return cleaned_data


def Preprocess_risk_level(df):
    """
    Preprocesses the risk level data by merging the the risk levels into three levels and four levels, and change the levels to start from 0.

    Args:
      df (pandas.DataFrame): The input data containing the risk level column.

    Returns:
      pandas.DataFrame: The processed data with additional columns for merged risk levels.
    """
    # 三個等級
    df["risk3"] = df["risk_level"].apply(
        lambda x: "0" if x in ["1", "2"] else ("1" if x == "3" else "2")
    )
    # 四個等級
    df["risk4"] = df["risk_level"].apply(
        lambda x: "0" if x == "1" else ("1" if x == "2" else ("2" if x == "3" else "3"))
    )
    # 五個等級
    df["risk5"] = df["risk_level"].apply(lambda x: str(int(x) - 1))

    return df


def Preprocess_transcripts(data):
    """
    Preprocesses the transcripts data by adding 'dialogue_id'、'utterance_id'、'utterance_len' columns to each transcript.

    Args:
      data (dict): A dictionary containing the transcripts data.

    Returns:
      dict: A dictionary with the preprocessed transcripts data.

    """
    preprocessed_data = data.copy()

    for key, value in preprocessed_data.items():
        value.insert(
            0, "dialogue_id", key
        )  # Add a column 'dialogue_id' to each transcript, with the value being the filename of the transcript
        value.insert(
            1, "utterance_id", range(1, 1 + len(value))
        )  # Add a column 'utterance_id' to each transcript, with the value being the index of the utterance
        value.insert(
            2, "utterance_len", value["utterance"].apply(lambda x: len(x))
        )  # Add a column 'utterance_len' to each transcript, with the value being the number of characters in the utterance

    return preprocessed_data


def Calc_consine_sim(emb1, emb2):
    """
    Calculate the cosine similarity between two embeddings.

    Args:
      emb1 (numpy.ndarray): The first embedding.
      emb2 (numpy.ndarray): The second embedding.

    Returns:
      float: The cosine similarity between the two embeddings.

    """
    return float(sentence_transformers.util.pytorch_cos_sim(emb1, emb2))


def Calc_max_consine_sim(emb1, emb2_li):
    """
    Calculate the maximum cosine similarity between an embedding and a list of embeddings.

    Args:
      emb1 (numpy.ndarray): The first embedding.
      emb2_li (list): A list of embeddings.

    Returns:
      float: The maximum cosine similarity between the first embedding and the list of embeddings.

    """
    return max([Calc_consine_sim(emb1, emb2) for emb2 in emb2_li])


def Calc_avg_consine_sim(emb1, emb2_li):
    """
    Calculate the average cosine similarity between an embedding and a list of embeddings.

    Args:
      emb1 (numpy.ndarray): The first embedding.
      emb2_li (list): A list of embeddings.

    Returns:
      float: The average cosine similarity between the first embedding and the list of embeddings.

    """
    return sum([Calc_consine_sim(emb1, emb2) for emb2 in emb2_li]) / len(emb2_li)


def Save_data(df, folder_path, file_name):
    """
    Save the given data to a CSV file.

    Args:
      df (pandas.DataFrame): The data to be saved.
      folder_path (str): The path to the folder where the file will be saved.
      file_name (str): The name of the file.

    Returns:
      None
    """
    df.to_csv(folder_path + file_name, index=False)


def main():
    """
    Preprocesses transcripts, risk levels, and reference sentences.
    Calculates sentence embeddings and similarities between transcripts and reference sentences.
    Saves the processed data and embeddings.

    Returns:
        None
    """
    total_time0 = time.time()

    lifeline_path = "data/rawdata/lifeline985/"
    processed_path = "data/processed/"
    refSentences_path = "data/rawdata/ref_sentences/"

    # 自殺風險等級
    inputRiskLevel_df = Read_risk_level(lifeline_path)  # 讀取自殺風險等級
    cleanedRiskLevel_df = Clean_risk_level(inputRiskLevel_df)  # 清理自殺風險等級
    processedRiskLevel_df = Preprocess_risk_level(
        cleanedRiskLevel_df
    )  # 預處理自殺風險等級
    Save_data(
        processedRiskLevel_df, processed_path, "risklevel.csv"
    )  # 儲存自殺風險等級

    # 逐字稿
    inputTranscript_dict = Read_transcripts(lifeline_path)  # 讀取逐字稿
    cleanedTranscript_dict = Clean_transcripts(inputTranscript_dict)  # 清理逐字稿
    filteredTranscript_dict = dict(  # 篩選自殺等級為 1-5 的逐字稿
        filter(
            lambda item: item[0]
            in processedRiskLevel_df["dialogue_id"].astype(str).tolist(),
            cleanedTranscript_dict.items(),
        )
    )
    proccesedTranscript_dict = Preprocess_transcripts(
        filteredTranscript_dict
    )  # 預處理逐字稿
    transcripts_df = pd.concat(
        proccesedTranscript_dict.values(), ignore_index=True
    )  # 合併逐字稿
    Save_data(transcripts_df, processed_path, "transcripts.csv")  # 儲存逐字稿

    print("Preprocessed transcripts and risk levels.")

    # 參考句子
    refSentencesCsv_li = [
        "BSS_suicidalRisk_scale.csv",
        "BSS_scale.csv",
        "suicidalRisk_scale.csv",
        "Lv_suicide_dict.csv",
    ]  # 合併量表、自殺意念量表、自殺危險程度量表、自殺辭典
    refSentences1_df = pd.read_csv(
        refSentences_path + refSentencesCsv_li[0], encoding="utf-8", header=None
    )
    refSentences2_df = pd.read_csv(
        refSentences_path + refSentencesCsv_li[1], encoding="utf-8", header=None
    )
    refSentences3_df = pd.read_csv(
        refSentences_path + refSentencesCsv_li[2], encoding="utf-8", header=None
    )
    refSentences4_df = pd.read_csv(
        refSentences_path + refSentencesCsv_li[3], encoding="utf-8", header=None
    )
    refSentences1_df.columns = ["text"]
    refSentences2_df.columns = ["text"]
    refSentences3_df.columns = ["text"]
    refSentences4_df.columns = ["text"]

    # Sentence embedding model
    model = sentence_transformers.SentenceTransformer(
        "paraphrase-xlm-r-multilingual-v1"
    )  # sentence-transformers 的模型
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device) # 不用因為沒有比較快

    transcripts_df["emb"] = list(model.encode(transcripts_df["utterance"]))
    refSentences1_df["emb"] = list(model.encode(refSentences1_df["text"].tolist()))
    refSentences2_df["emb"] = list(model.encode(refSentences2_df["text"].tolist()))
    refSentences3_df["emb"] = list(model.encode(refSentences3_df["text"].tolist()))
    refSentences4_df["emb"] = list(model.encode(refSentences4_df["text"].tolist()))

    refSentences1_df.to_csv(processed_path + "refSentences1_emb.csv", index=False)
    refSentences2_df.to_csv(processed_path + "refSentences2_emb.csv", index=False)
    refSentences3_df.to_csv(processed_path + "refSentences3_emb.csv", index=False)
    refSentences4_df.to_csv(processed_path + "refSentences4_emb.csv", index=False)

    print("Added sentence embeddings.")

    print("\nCalculating similarities...")

    time0 = time.time()

    # 計算逐字稿語句和參照句子的相似度，並將其最大值／平均值作為與自殺的相關分數
    transcripts_df["max_sim_ref1"] = transcripts_df["emb"].apply(
        lambda x: Calc_max_consine_sim(x, refSentences1_df["emb"].tolist())
    )
    transcripts_df["avg_sim_ref1"] = transcripts_df["emb"].apply(
        lambda x: Calc_avg_consine_sim(x, refSentences1_df["emb"].tolist())
    )

    transcripts_df["max_sim_ref2"] = transcripts_df["emb"].apply(
        lambda x: Calc_max_consine_sim(x, refSentences2_df["emb"].tolist())
    )
    transcripts_df["avg_sim_ref2"] = transcripts_df["emb"].apply(
        lambda x: Calc_avg_consine_sim(x, refSentences2_df["emb"].tolist())
    )

    transcripts_df["max_sim_ref3"] = transcripts_df["emb"].apply(
        lambda x: Calc_max_consine_sim(x, refSentences3_df["emb"].tolist())
    )
    transcripts_df["avg_sim_ref3"] = transcripts_df["emb"].apply(
        lambda x: Calc_avg_consine_sim(x, refSentences3_df["emb"].tolist())
    )

    transcripts_df["max_sim_ref4"] = transcripts_df["emb"].apply(
        lambda x: Calc_max_consine_sim(x, refSentences4_df["emb"].tolist())
    )
    transcripts_df["avg_sim_ref4"] = transcripts_df["emb"].apply(
        lambda x: Calc_avg_consine_sim(x, refSentences4_df["emb"].tolist())
    )

    time1 = time.time()

    print("Calculate similarity time elapsed:", utils.Format_time(time1 - time0))

    Save_data(
        transcripts_df, processed_path, "transcripts_emb.csv"
    )  # 儲存逐字稿的嵌入和相似度

    # 分成只有求助者(c, C)、只有輔導員(r, R)、和兩者都有的對話儲存
    Save_data(
        transcripts_df[
            (transcripts_df["role"] == "c") | (transcripts_df["role"] == "C")
        ],
        processed_path,
        "transcripts_emb_caller.csv",
    )
    Save_data(
        transcripts_df[
            (transcripts_df["role"] == "r") | (transcripts_df["role"] == "R")
        ],
        processed_path,
        "transcripts_emb_responser.csv",
    )
    Save_data(
        transcripts_df[
            (transcripts_df["role"] == "c")
            | (transcripts_df["role"] == "C")
            | (transcripts_df["role"] == "r")
            | (transcripts_df["role"] == "R")
        ],
        processed_path,
        "transcripts_emb_all.csv",
    )

    total_time1 = time.time()
    print("Total time elapsed:", utils.Format_time(total_time1 - total_time0))


if __name__ == "__main__":
    main()
    print("Data processing completed.")
