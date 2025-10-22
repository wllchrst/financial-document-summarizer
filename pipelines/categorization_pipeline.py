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
    Extract the title portion from a file path.

    The title is expected to follow the format: "[<number>] <title-text>".
    Example:
        Input: "data/valid_processed/[1000000] General Information.csv"
        Output: "[1000000] General Information"

    Args:
        filepath (str): The full path of the file.

    Returns:
        str: The extracted title string.
    """
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    match = re.match(r"(\[\d+\]\s*.+)", name)
    return match.group(1) if match else name


def get_json_file(filepath: str):
    """
    Read and load a JSON file into Python data structure.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        Any: Parsed JSON content.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


def retrieve_data(filepath: Optional[str] = None,
                  code: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from a JSON file either by direct filepath or searching by code.

    If `filepath` is provided, it loads data from that file. Otherwise, it searches
    the PROCESSED_FOLDER directory for a file containing the given code.

    Args:
        filepath (Optional[str]): Direct path to a JSON file.
        code (Optional[str]): Code to search in filenames if filepath is not provided.

    Raises:
        FileNotFoundError: If code is not found in processed folder.

    Returns:
        dict: Parsed JSON content.
    """
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
    """
    Categorize the content of a financial table JSON file using LLM classification.

    Each table can be nested or flat. Nested tables are labeled recursively.

    Args:
        use_gemini (bool): Whether to use Gemini model or another model for classification.
        testing (bool): If True, stops inference early for faster testing.
        filepath (Optional[str]): Path to the input JSON file.
        code (Optional[str]): File code if filepath not provided.

    Raises:
        ValueError: If both filepath and code are None.

    Side Effects:
        Saves labeled JSON output into LABELED_PROCESSED_FOLDER.
    """
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
    """
    Recursively classify nested dictionary structures of table data.

    Args:
        data (dict): Nested table data.
        title (str): Title of the section being classified.
        descriptions (List[str]): Parent descriptions accumulated during recursion.
        prompter (CategorizePrompt): Classification model wrapper.
        testing (bool): Stops early when testing mode is active.

    Returns:
        dict: Updated nested data with labels embedded.
    """
    global inference_totalread

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
    """
    Classify non-nested (flat) tabular data row-wise.

    Args:
        data (list): List of row dictionaries with 'descriptions' and 'values'.
        title (str): Section title.
        prompter (CategorizePrompt): Classification model.
        testing (bool): If True, stops after two classifications.

    Returns:
        dict: Dictionary mapping value descriptions to assigned labels.
    """
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
