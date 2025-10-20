import pandas as pd
import os
import re
import json

from prompt.categorize_prompt import CategorizePrompt
from typing import *

PROCESSED_FOLDER = 'data/processed'
LABELED_PROCESSED_FOLDER = 'data/labeled_data'
inference_total = 0


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

    filenames = os.listdir(PROCESSED_FOLDER)
    for filename in filenames:
        if code in filename:
            path = os.path.join(PROCESSED_FOLDER, filename)
            return get_json_file(path)

    raise FileNotFoundError("Code is not found in valid processed folder")


def categorize_code(use_gemini: bool,
                    testing: bool,
                    filepath: Optional[str] = None,
                    code: Optional[str] = None) -> pd.DataFrame:
    if filepath is None and code is None:
        raise ValueError("filepath and code cannot be None")
    prompter = CategorizePrompt(use_gemini)
    os.makedirs(LABELED_PROCESSED_FOLDER, exist_ok=True)

    data = retrieve_data(filepath, code)
    title = data['title']

    for table in data['tables']:
        is_nested = table['nested_result']
        global inference_total
        inference_total = 0

        if is_nested:
            table['data'] = categorize_nested_data(data=table['data'], title=title, descriptions=[], prompter=prompter,
                                                   testing=testing)
        else:
            labels = categorize_data(data=table['data'], title=title, prompter=prompter, testing=testing)
            table['labels'] = labels

    save_path = os.path.join(LABELED_PROCESSED_FOLDER, f'{title}.json')
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def categorize_nested_data(data,
                           title: str,
                           descriptions: List[str],
                           prompter: CategorizePrompt,
                           testing: bool):
    global inference_total

    if inference_total == 2 and testing:
        return data

    for key, value in data.items():

        if isinstance(value, dict):
            descriptions.append(key)
            categorize_nested_data(value, title, descriptions, prompter, testing)
            descriptions.pop()

        elif isinstance(value, str) and value.strip() != '':
            general_label, detailed_label = prompter.classify(title, descriptions)
            data[key] = {
                "value": value,
                "general_label": general_label,
                "detailed_label": detailed_label
            }

            inference_total += 1

    return data


def categorize_data(data,
                    title: str,
                    prompter: CategorizePrompt,
                    testing: bool):
    global inference_total
    if len(data) == 0:
        return data

    header_descriptions = list(data[0]['descriptions'].values())
    value_descriptions = list(data[0]['values'].keys())
    labels = {}

    for desc in value_descriptions:
        general_label, detailed_label = prompter.classify(
            title=title,
            descriptions=header_descriptions + [desc]
        )

        labels[desc] = {
            'general_label': general_label,
            'detailed_label': detailed_label
        }

        inference_total += 1
        if inference_total == 2 and testing:
            break

    return labels
