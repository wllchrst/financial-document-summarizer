import pandas as pd
import os
import re

from prompt.categorize_prompt import CategorizePrompt
from typing import *

VALID_PROCESSED_FOLDER = 'data/valid_processed'
LABELED_PROCESSED_FOLDER = 'data/labeled_data'


def extract_title_from_path(filepath: str) -> str:
    """
    Extracts the title (e.g. "[1000000] General Information")
    from a file path like 'data/valid_processed/[1000000] General Information.csv'
    """
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    match = re.match(r"(\[\d+\]\s*.+)", name)
    return match.group(1) if match else name


def retrieve_data(filepath: Optional[str] = None,
                  code: Optional[str] = None) -> pd.DataFrame:
    if filepath is not None:
        df = pd.read_csv(filepath)
        return df, extract_title_from_path(filepath)

    filenames = os.listdir(VALID_PROCESSED_FOLDER)
    for filename in filenames:
        if code in filename:
            path = os.path.join(VALID_PROCESSED_FOLDER, filename)
            df = pd.read_csv(path)
            return df, filename.split('.')[0]

    raise FileNotFoundError("Code is not found in valid processed folder")


def categorize_data(use_gemini: bool,
                    filepath: Optional[str] = None,
                    code: Optional[str] = None) -> pd.DataFrame:
    if filepath is None and code is None:
        raise ValueError("filepath and code cannot be None")

    prompter = CategorizePrompt(use_gemini)

    df, title = retrieve_data(filepath, code)
    cols = df.columns
    desc_cols = [col for col in cols if 'desc' in col]
    general_labels = []
    detailed_labels = []

    for index, row in df.iterrows():
        row_descriptions = [row[col] for col in desc_cols]
        general_label, detailed_label = prompter.classify(
            descriptions=row_descriptions,
            title=title
        )
        general_labels.append(general_label)
        detailed_labels.append(detailed_label)

    df['General Label'] = general_labels
    df['Detailed Label'] = detailed_labels

    os.makedirs(LABELED_PROCESSED_FOLDER, exist_ok=True)
    save_path = os.path.join(LABELED_PROCESSED_FOLDER, f'{title}.csv')

    print(f"Dataframe saved in {save_path}")
    df.to_csv(save_path)
    return df
