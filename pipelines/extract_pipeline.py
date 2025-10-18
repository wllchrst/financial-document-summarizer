import os
import pdfplumber
import pandas as pd
import cv2
import numpy as np

from typing import *
from helpers import WordHelper
from pdfplumber.page import Page
from pdf2image import convert_from_path


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
        page_indices = []
        current_code = ''

        for index, page in enumerate(pdf.pages):
            content = page.extract_text()
            code_in_content = None

            for code in codes:
                if code in content:
                    code_in_content = code
                    break

            if not found_starting and code_in_content:
                found_starting = True
                current_code = code_in_content
                pages.append(page)
                page_indices.append(index)


            elif found_starting and not code_in_content:
                pages.append(page)
                page_indices.append(index)

            elif found_starting and code_in_content:
                extraction_result = extract_bilingual_lines(
                    pages=pages,
                    code=current_code,
                    y_threshold=3,
                    filepath=filepath,
                    page_indices=page_indices
                )

                extractions[current_code] = extraction_result

                pages = [page]
                page_indices = [index]
                current_code = code_in_content

        if pages and current_code not in extractions:
            extraction_result = extract_bilingual_lines(
                pages=pages,
                code=current_code,
                y_threshold=3,
                filepath=filepath,
                page_indices=page_indices
            )
            extractions[current_code] = extraction_result

    return extractions


def extract_bilingual_lines(pages: List[Page],
                            code: str,
                            filepath: str,
                            page_indices: List[int],
                            y_threshold=3,
                            gap_threshold=4):
    """
    Groups words by line, then splits them into multiple parts
    based on horizontal gaps (e.g. left + right bilingual text).
    For each cluster, stores both text and rounded height (max - min vertical range).
    """
    if '1410000' not in code:
        return
    final_data = []
    max_height = 0
    min_height = float('inf')

    for page_idx, page in enumerate(pages, start=1):
        data = []
        rects = page.rects
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

            current_cluster = [items[0]]

            for i in range(1, len(items)):
                prev_x = items[i - 1]["x1"]
                curr_x = items[i]["x0"]
                gap = curr_x - prev_x

                if gap > gap_threshold:
                    top_vals = [w["top"] for w in current_cluster]
                    bottom_vals = [w["bottom"] for w in current_cluster]
                    cluster_height = round(max(bottom_vals) - min(top_vals))
                    max_height = max(max_height, cluster_height)
                    min_height = min(min_height, cluster_height)

                    clusters.append(make_cluster(current_cluster, height=cluster_height))
                    current_cluster = [items[i]]
                else:
                    current_cluster.append(items[i])

            top_vals = [w["top"] for w in current_cluster]
            bottom_vals = [w["bottom"] for w in current_cluster]
            cluster_height = round(max(bottom_vals) - min(top_vals))
            max_height = max(max_height, cluster_height)
            min_height = min(min_height, cluster_height)

            clusters.append(make_cluster(current_cluster, height=cluster_height))
            data.append(clusters)

        final_grouping = group_vertically(data, rects, max_height=max_height, min_height=min_height)

    return 1


def make_cluster(words, height):
    top = words[0]['top']
    bottom = words[0]['bottom']
    x0, x1 = words[0]['x0'], words[0]['x1']

    return {
        'text': " ".join(w['text'] for w in words),
        'height': height,
        'top': top,
        'bottom': bottom,
        'x0': x0,
        'x1': x1,
    }


def group_vertically(clusters,
                     rects,
                     max_height: int,
                     min_height: int):
    """
    Group clusters vertically so that every table row consists of 2 lines,
    separated by horizontal rectangles.
    """

    rects = [rect for rect in rects if rect['height'] < 2]
    rects = sorted(rects, key=lambda r: r['y0'])
    title = ''
    top_columns = []
    value_columns = []

    for index, cluster in enumerate(index, clusters):
        x = 0
        if len(cluster) == 1 and cluster[0]['height'] == max_height:
            title += cluster[0]['text']
        elif cluster[0]['height'] == max_height:
            for index, col in enumerate(cluster):
                if len(top_columns) != len(cluster):
                    top_columns.append(col['text'])
                else:
                    top_columns[index] += col['text']
        elif cluster[0]['height'] == min_height:
            value_columns = manage_value_columns(clusters, index)

    for index in range(1, len(rects)):
        prev_line = rects[index - 1]
        next_line = rects[index]

        if prev_line['y0'] == next_line['y0']:
            continue

        group_cluster = []

        for cluster in clusters:
            cluster_top = cluster[0]['top']
            cluster_bottom = cluster[0]['bottom']
            if cluster_top > prev_line['y1'] and cluster_bottom < next_line['y0']:
                group_cluster.append(cluster)

        if len(group_cluster) > 0:
            lm = 0

def manage_value_columns(clusters, index: int, min_height: int):


def clean_extraction_result():
    pass
