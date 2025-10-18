import os
import pdfplumber
import pandas as pd
import numpy as np

from typing import *
from helpers import WordHelper
from pdfplumber.page import Page

PROCESSED_FOLDER_PATH = 'data/processed'


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


import os
import pdfplumber


def extract_pdf(filepath: str):
    os.makedirs(PROCESSED_FOLDER_PATH, exist_ok=True)
    codes = get_title_codes()
    extractions = {}

    def save_and_store(pages, page_indices, code):
        extraction_result, title = extract_bilingual_lines(
            pages=pages,
            code=code,
            y_threshold=3,
            filepath=filepath,
            page_indices=page_indices
        )
        if extraction_result is not None:
            extraction_result.to_csv(f"{PROCESSED_FOLDER_PATH}/{title}.csv")
        extractions[code] = extraction_result

    with pdfplumber.open(filepath) as pdf:
        pages, page_indices, current_code = [], [], None

        for index, page in enumerate(pdf.pages):
            content = page.extract_text() or ""
            code_found = next((code for code in codes if code in content), None)

            if code_found:
                if current_code:  # Save previous block
                    save_and_store(pages, page_indices, current_code)
                # Start new block
                pages, page_indices = [page], [index]
                current_code = code_found
            elif current_code:
                pages.append(page)
                page_indices.append(index)

        # Process last remaining block
        if current_code and current_code not in extractions:
            save_and_store(pages, page_indices, current_code)

    return extractions


def extract_bilingual_lines(pages: List[Page],
                            code: str,
                            filepath: str,
                            page_indices: List[int],
                            y_threshold=3,
                            gap_threshold=4) -> Tuple[pd.DataFrame, str]:
    """
    Groups words by line, then splits them into multiple parts
    based on horizontal gaps (e.g. left + right bilingual text).
    For each cluster, stores both text and rounded height (max - min vertical range).
    """
    final_data = []
    max_height = 0
    min_height = float('inf')
    main_page_parsed = False

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

        if main_page_parsed:
            _, _, _, rows = group_vertically(data, rects, max_height=max_height,
                                             min_height=min_height,
                                             main_page_parsed=main_page_parsed)
        else:
            title, top_columns, value_columns, rows = group_vertically(data, rects, max_height=max_height,
                                                                       min_height=min_height,
                                                                       main_page_parsed=main_page_parsed)

        final_result = format_final_result(top_columns=top_columns, value_columns=value_columns, rows=rows)
        main_page_parsed = True
        final_data += final_result

    df = format_into_df(final_data, top_columns, value_columns)
    return df, title


def make_cluster(words, height):
    top = words[0]['top']
    bottom = words[0]['bottom']
    x0 = min(w['x0'] for w in words)
    x1 = max(w['x1'] for w in words)
    center_x = round((x0 + x1) / 2)

    return {
        'text': " ".join(w['text'] for w in words),
        'height': height,
        'top': top,
        'bottom': bottom,
        'x': center_x,
        'min_x': x0,
        'max_x': x1
    }


def group_vertically(clusters,
                     rects,
                     max_height: int,
                     min_height: int,
                     main_page_parsed: bool):
    """
    Group clusters vertically so that every table row consists of 2 lines,
    separated by horizontal rectangles.
    """
    row_rects = [rect for rect in rects if rect['height'] < 2]
    row_rects = sorted(row_rects, key=lambda r: r['top'])
    col_rects = [rect for rect in rects if rect['height'] > 10]
    title = ''
    top_columns = []
    value_columns = []
    index = 0

    if not main_page_parsed:
        for index, cluster in enumerate(clusters):
            x = 0
            if len(cluster) == 1 and cluster[0]['height'] == max_height:
                title += cluster[0]['text']
            elif cluster[0]['height'] == max_height:
                top_columns, index = manage_top_columns(clusters, index, max_height)
            elif cluster[0]['height'] == min_height:
                value_columns, index = manage_value_columns(clusters, index, min_height, col_rects)
                break

    clusters = clusters[index:]
    rows = retrieve_data_rows(clusters, row_rects)
    return title, top_columns, value_columns, rows


def retrieve_data_rows(clusters, rects):
    rows = []

    for i in range(1, len(rects)):
        prev_line = rects[i - 1]
        next_line = rects[i]

        row_clusters = []
        for cluster in clusters:
            cluster_top = cluster[0]['top']
            cluster_bottom = cluster[0]['bottom']

            if cluster_bottom > prev_line['bottom'] and cluster_top < next_line['top']:
                row_clusters += cluster

        if row_clusters:
            rows.append(row_clusters)

    return rows


def manage_value_columns(clusters, starting_index: int, min_height: int, col_rects, tolerance: int = 3):
    values = {}

    for index in range(starting_index, len(clusters)):
        cluster = clusters[index]
        if cluster[0]['height'] != min_height:
            break

        for value in cluster:
            x = value['x']
            matched_key = None
            for key in values:
                if abs(key - x) <= tolerance:
                    matched_key = key
                    break

            if matched_key is not None:
                values[matched_key]["text"] += " " + value["text"]
            else:
                values[x] = {
                    "text": value["text"],
                    "x": value['x'],
                    'min_x': value['min_x'],
                    'max_x': value['max_x']
                }

    return [values[k] for k in sorted(values)], index


def manage_top_columns(clusters, starting_index: int, max_height: int):
    top_columns = []
    for i in range(starting_index, len(clusters)):
        cluster = clusters[i]

        if cluster[0]['height'] != max_height:
            break

        for index, col in enumerate(cluster):
            if len(top_columns) != len(cluster):
                top_columns.append({
                    'text': col['text'],
                    'x': col['x'],
                    'min_x': col['min_x'],
                    'max_x': col['max_x']
                })
            else:
                top_columns[index]['text'] += f' {col['text']}'
                top_columns[index]['min_x'] = min(top_columns[index]['min_x'], col['min_x'])
                top_columns[index]['max_x'] = max(top_columns[index]['max_x'], col['max_x'])

    return top_columns, i


def format_final_result(top_columns, value_columns, rows,
                        tolerance_for_value=10, tolerance_for_top=3):
    results = []

    all_columns = []
    for i, col in enumerate(value_columns):
        all_columns.append({"index": i, "type": "value", **col})
    for i, col in enumerate(top_columns):
        all_columns.append({"index": i, "type": "top", **col})

    for cluster in rows:
        row_data = {
            "values": {i: "" for i in range(len(value_columns))},
            "top": {i: "" for i in range(len(top_columns))}
        }

        if len(results) == 3:
            lm = 1

        sorted_cluster = sorted(cluster, key=lambda w: (round(w["top"], 1), w["min_x"]))

        for word in sorted_cluster:
            x_center = (word["min_x"] + word["max_x"]) / 2
            matched_col = None

            for col in all_columns:
                tolerance = tolerance_for_top if col['type'] == 'top' else tolerance_for_value
                if col["min_x"] - tolerance <= x_center <= col["max_x"] + tolerance:
                    matched_col = col
                    break

            if matched_col:
                if matched_col["type"] == "top":
                    row_data["top"][matched_col["index"]] = (
                            row_data["top"][matched_col["index"]] + " " + word["text"]
                    ).strip()
                else:
                    row_data["values"][matched_col["index"]] = (
                            row_data["values"][matched_col["index"]] + " " + word["text"]
                    ).strip()

        results.append(row_data)

    return results


def format_into_df(parsed_rows, top_columns, value_columns) -> pd.DataFrame:
    table_rows = []

    for row in parsed_rows:
        flat_row = {}

        for i, col in enumerate(top_columns):
            flat_row[col["text"] + '_desc'] = row["top"].get(i, "")

        for i, col in enumerate(value_columns):
            flat_row[col["text"] + '_value'] = row["values"].get(i, "")

        table_rows.append(flat_row)

    return pd.DataFrame(table_rows)
