import os
import pdfplumber
import pandas as pd

from typing import *
from helpers import WordHelper


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
    table_dictionary = {}
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            print('_' * 50)
            table = page.extract_table()
            columns = table[0]
            rows = table[1:]
            dataframe = pd.DataFrame(rows, columns=columns)

            print(f'Columns: {columns}')
            print(dataframe.head(20))


def extract_table_from_pdf(filepath: str):
    with pdfplumber.open(filepath) as pdf:

        for index, page in enumerate(pdf.pages):
            if index != 1:
                continue
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            lines = {}

            for w in words:
                y = round(w["top"], 1)
                lines.setdefault(y, []).append(w)

            print(lines)
            break
