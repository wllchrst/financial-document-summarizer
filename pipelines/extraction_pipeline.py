import os
import pdfplumber
import pandas as pd
import json
import re

from typing import *
from helpers import WordHelper
from pdfplumber.page import Page

PROCESSED_FOLDER_PATH = 'data/processed'
PROCESSED_FOLDER_PATH_JSON = 'data/processed_json'


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
        if code_name in codes:
            continue

        description = ''
        if 'BD' in code_name:
            description = '(breakdown)'
        elif 'PY' in code_name:
            description = 'Prior Year'

        codes.append({
            'filtered': WordHelper.filter_numeric(code_name),
            'raw': code_name,
            'description': description
        })

    return codes


def extract_pdf(filepath: str):
    os.makedirs(PROCESSED_FOLDER_PATH, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER_PATH_JSON, exist_ok=True)
    codes = get_title_codes()
    extractions = {}

    def save_and_store(content_code: str):
        dataframe, title, descriptions = extract_bilingual_lines(
            pages=pages,
            y_threshold=3,
        )
        extraction_result = None

        if dataframe is not None:
            match = re.search(r'\[(.*?)\]', title)
            c = match.group(1)
            py_information = '' if 'prior year' not in title.lower() else '_PriorYear'
            filename = c + py_information

            dataframe.to_csv(f"{PROCESSED_FOLDER_PATH}/{filename}.csv", index=False)
            data_records = dataframe.to_dict(orient="records")

            extraction_result = {
                "code": content_code,
                "title": title,
                "descriptions": descriptions,
                "data": data_records
            }

            with open(f"{PROCESSED_FOLDER_PATH_JSON}/{filename}.json", "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)

        extractions[content_code] = extraction_result

    with pdfplumber.open(filepath) as pdf:
        pages, page_indices, current_code = [], [], None

        for index, page in enumerate(pdf.pages):
            content = page.extract_text() or ""
            code_found = None

            for code in codes:
                filtered = code['filtered']
                raw = code['raw']
                if filtered in content and code['description'] in content:
                    code_found = raw

            if code_found:
                if current_code:
                    save_and_store(content_code=current_code)
                pages, page_indices = [page], [index]
                current_code = code_found
            elif current_code:
                pages.append(page)
                page_indices.append(index)

        if current_code and current_code not in extractions:
            save_and_store(content_code=current_code)

    return extractions


def extract_bilingual_lines(pages: List[Page],
                            y_threshold=3,
                            gap_threshold=4) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Groups words by line, then splits them into multiple parts
    based on horizontal gaps (e.g. left + right bilingual text).
    For each cluster, stores both text and rounded height (max - min vertical range).
    """
    # if "10000" not in pages[0].extract_text():
    #     return None, None, None

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
            _, _, _, _, rows = group_vertically(data, rects, max_height=max_height,
                                                min_height=min_height,
                                                main_page_parsed=main_page_parsed)
        else:
            title, descriptions, column_barriers, value_columns, rows = group_vertically(data, rects,
                                                                                         max_height=max_height,
                                                                                         min_height=min_height,
                                                                                         main_page_parsed=main_page_parsed)

        final_result = format_final_result(column_barriers=column_barriers, value_columns=value_columns, rows=rows)
        main_page_parsed = True
        final_data += final_result

    df = format_into_df(final_data, desc_columns=column_barriers, value_columns=value_columns)
    return df, title, descriptions


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
    descriptions = []
    value_columns = []
    index = 0

    if not main_page_parsed:
        for index, cluster in enumerate(clusters):
            if len(cluster) == 1 and cluster[0]['height'] == max_height:
                title += cluster[0]['text']
            elif cluster[0]['height'] == max_height:
                descriptions, index = manage_descriptions(clusters, index, max_height)
            elif cluster[0]['height'] == min_height:
                value_columns, index = manage_value_columns(clusters, index, min_height, col_rects)
                break

    clusters = clusters[index:]
    column_barriers = get_description_column_barriers(clusters, value_columns)
    rows = retrieve_data_rows(clusters, row_rects)
    return title, descriptions, column_barriers, value_columns, rows


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


def manage_descriptions(clusters, starting_index: int, max_height: int) -> List[str]:
    descriptions = []
    for i in range(starting_index, len(clusters)):
        cluster = clusters[i]

        if cluster[0]['height'] != max_height:
            break

        for index, sentence in enumerate(cluster):
            if len(descriptions) != len(cluster):
                descriptions.append(sentence['text'])
            else:
                descriptions[index] += sentence['text']

    return descriptions, i


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


def get_description_column_barriers(
        clusters,
        value_columns,
        merge_tolerance: float = 5.0
):
    """
    Identify valid description column barriers based on both spatial layout
    and content presence across rows.

    A barrier is kept if:
      - It has content in every row, OR
      - It has content in the first row only.

    Args:
        clusters (List[List[dict]]): List of line clusters; each cluster represents one row.
        value_columns (List[dict]): List of value column boundaries.
        merge_tolerance (float): Distance threshold to merge nearby barriers.

    Returns:
        List[dict]: Filtered barriers with 'barrier' (min_x, max_x) and 'col_name'.
    """

    def is_inside_value_col(center_x: float) -> bool:
        """Return True if a point lies within any value column range."""
        return any(v['min_x'] <= center_x <= v['max_x'] for v in value_columns)

    candidates = []
    for cluster in clusters:
        for sentence in cluster:
            center_x = sentence['x']
            if not is_inside_value_col(center_x):
                candidates.append((sentence['min_x'], sentence['max_x']))

    if not candidates:
        return []

    candidates.sort(key=lambda b: b[0])
    merged = []
    current_min, current_max = candidates[0]

    for min_x, max_x in candidates[1:]:
        if min_x - current_max <= merge_tolerance:
            current_max = max(current_max, max_x)
        else:
            merged.append((current_min, current_max))
            current_min, current_max = min_x, max_x
    merged.append((current_min, current_max))

    valid_barriers = []
    total_rows = len(clusters)

    for barrier_idx, (bar_min, bar_max) in enumerate(merged):
        rows_with_content = 0
        for cluster_idx, cluster in enumerate(clusters):
            for sentence in cluster:
                center_x = sentence['x']
                if bar_min <= center_x <= bar_max:
                    rows_with_content += 1
                    break

        filled_everywhere = rows_with_content == total_rows
        filled_first_row = any(
            bar_min <= s['x'] <= bar_max for s in clusters[0]
        )

        if filled_everywhere or filled_first_row:
            valid_barriers.append({
                'barrier': (bar_min, bar_max),
                'col_name': f'desc_col_{len(valid_barriers)}',
                'fill_ratio': rows_with_content / total_rows
            })

    return valid_barriers


def format_final_result(column_barriers, value_columns, rows,
                        tolerance_for_value=10, tolerance_for_top=3):
    results = []

    value_cols, desc_cols = [], []
    for i, col in enumerate(value_columns):
        value_cols.append({"index": i, "type": "value", **col})
    for i, col in enumerate(column_barriers):
        desc_cols.append({"index": i, "type": "top", **col})

    for cluster in rows:
        row_data = {
            "values": {i: "" for i in range(len(value_columns))},
            "descriptions": {i: "" for i in range(len(column_barriers))}
        }

        sorted_cluster = sorted(cluster, key=lambda w: (round(w["top"], 1), w["min_x"]))

        for word in sorted_cluster:
            x_center = (word["min_x"] + word["max_x"]) / 2
            matched_col = None

            for col in value_cols:
                tolerance = tolerance_for_top if col['type'] == 'descriptions' else tolerance_for_value
                if col["min_x"] - tolerance <= x_center <= col["max_x"] + tolerance:
                    matched_col = col
                    break

            for col in desc_cols:
                tolerance = tolerance_for_top if col['type'] == 'descriptions' else tolerance_for_value
                barrier = col['barrier']
                if barrier[0] - tolerance <= x_center <= barrier[1] + tolerance:
                    matched_col = col
                    break

            if matched_col:
                if matched_col["type"] == "top":
                    row_data["descriptions"][matched_col["index"]] = (
                            row_data["descriptions"][matched_col["index"]] + " " + word["text"]
                    ).strip()
                else:
                    row_data["values"][matched_col["index"]] = (
                            row_data["values"][matched_col["index"]] + " " + word["text"]
                    ).strip()

        results.append(row_data)

    return results


def format_into_df(parsed_rows, desc_columns, value_columns) -> pd.DataFrame:
    table_rows = []

    for row in parsed_rows:
        flat_row = {}

        for i, col in enumerate(desc_columns):
            flat_row[col["col_name"]] = row["descriptions"].get(i, "")

        for i, col in enumerate(value_columns):
            flat_row[col["text"] + '_value'] = row["values"].get(i, "")

        table_rows.append(flat_row)

    return pd.DataFrame(table_rows)
