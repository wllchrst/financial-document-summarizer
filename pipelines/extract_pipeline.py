import os
import pdfplumber
import pandas as pd

from typing import *
from helpers import WordHelper
from pdfplumber.page import Page


def get_title_codes(folder_name: str = 'data/inlineXBRL') -> List[str]:
    """
    Get every title code from the html (the html is retrieved from the website IDX the folder name is called inlineXBRL)

    :param folder_name: where is the file containing all code of the html
    :return: list of codes retrieved from folder
    """
    codes = []
    filenames = os.listdir(folder_name)
    for filename in filenames:
        code_name = filename.split('.')[0]
        code_name = WordHelper.filter_numeric(code_name)
        if code_name in codes:
            continue

        codes.append(code_name)

    return codes


def read_pdf(filepath: str):
    extractions = {}
    codes = get_title_codes()

    with pdfplumber.open(filepath) as pdf:
        found_starting = False
        pages = []
        current_code = ''

        for index, page in enumerate(pdf.pages):
            content = page.extract_text()
            code_in_content = False

            for code in codes:
                if code in content:
                    current_code = code
                    code_in_content = True
                    break

            if not found_starting and code_in_content:
                found_starting = True
                pages.append(page)
            elif found_starting and not code_in_content:
                pages.append(page)
            elif found_starting and code_in_content:
                extraction_result = extract_bilingual_lines(
                    pages=pages,
                    y_threshold=3,
                    gap_threshold=40
                )

                extractions[current_code] = extraction_result
                pages = []
                if len(extractions) == 2:
                    break


def extract_bilingual_lines(pages, y_threshold=3, gap_threshold=15):
    """
    Groups words by line, then splits them into multiple parts
    based on horizontal gaps (e.g. left + right bilingual text).
    """
    print(len(pages))
    data = []
    for page_idx, page in enumerate(pages, start=1):
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        lines = {}

        for w in words:
            y = round(w["top"], 1)
            matched_y = None
            for existing_y in lines.keys():
                if abs(existing_y - y) < y_threshold:
                    matched_y = existing_y
                    break

            if matched_y is not None:
                lines[matched_y].append(w)
            else:
                lines[y] = [w]

        for y, items in sorted(lines.items()):
            items.sort(key=lambda x: x["x0"])
            clusters = []
            current_cluster = [items[0]["text"]]

            for i in range(1, len(items)):
                prev_x = items[i - 1]["x1"]
                curr_x = items[i]["x0"]
                gap = curr_x - prev_x

                if gap > gap_threshold:
                    clusters.append(" ".join(current_cluster))
                    current_cluster = [items[i]["text"]]
                else:
                    current_cluster.append(items[i]["text"])

            clusters.append(" ".join(current_cluster))
            data.append(clusters)

    return data


def clean_extraction_result(data):
    print('#' * 100)
    for d in data:
        print(d)
