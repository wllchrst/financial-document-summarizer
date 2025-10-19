import pandas as pd
import os
import re
import json

from prompt.categorize_prompt import CategorizePrompt
from typing import *

PROCESSED_FOLDER_JSON = 'data/processed_json'
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


def get_json_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


def retrieve_data(filepath: Optional[str] = None,
                  code: Optional[str] = None) -> pd.DataFrame:
    if filepath is not None:
        return get_json_file(filepath)

    filenames = os.listdir(PROCESSED_FOLDER_JSON)
    for filename in filenames:
        if code in filename:
            path = os.path.join(PROCESSED_FOLDER_JSON, filename)
            return get_json_file(path)

    raise FileNotFoundError("Code is not found in valid processed folder")


def categorize_data(use_gemini: bool,
                    filepath: Optional[str] = None,
                    code: Optional[str] = None) -> pd.DataFrame:
    if filepath is None and code is None:
        raise ValueError("filepath and code cannot be None")

    prompter = CategorizePrompt(use_gemini)

    data = retrieve_data(filepath, code)
    df = pd.DataFrame(data['data'])
    title = data['title']
    header_descriptions = data['descriptions']

    cols = df.columns
    desc_cols = [col for col in cols if 'desc' in col]
    general_labels = []
    detailed_labels = []

    for index, row in df.iterrows():
        row_descriptions = [row[col] for col in desc_cols]
        general_label, detailed_label = prompter.classify(
            descriptions=row_descriptions,
            title=title,
            header_descriptions=header_descriptions
        )
        general_labels.append(general_label)
        detailed_labels.append(detailed_label)

    df = df[:index + 1]
    df['General Label'] = general_labels
    df['Detailed Label'] = detailed_labels

    os.makedirs(LABELED_PROCESSED_FOLDER, exist_ok=True)
    save_path = os.path.join(LABELED_PROCESSED_FOLDER, f'{title}.json')

    data['data'] = df.to_dict(orient='records')
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Json saved in {save_path}")

    return data
