import os
import pdfplumber
import pandas as pd
import json
import re

from typing import *
from helpers import WordHelper
from pdfplumber.page import Page

PROCESSED_FOLDER_PATH = 'data/processed'


def get_title_codes(folder_name: str = 'data/inlineXBRL') -> List[dict]:
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
    codes = get_title_codes()
    extractions = {}

    def save_and_store(content_code: str):
        title, tables = extract_all_data(
            pages=pages,
            y_threshold=3,
        )
        extraction_result = None

        if tables is None:
            extractions[content_code] = extraction_result
            return

        match = re.search(r'\[(.*?)\]', title)
        c = match.group(1)
        py_information = '' if 'prior year' not in title.lower() else '_PriorYear'
        filename = c + py_information

        extraction_result = {
            'title': title,
            'content_code': code,
            'tables': tables
        }

        with open(f"{PROCESSED_FOLDER_PATH}/{filename}.json", "w", encoding="utf-8") as f:
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


def extract_all_data(pages: List[Page],
                     y_threshold=3,
                     big_threshold=4,
                     medium_threshold=3,
                     small_threshold=2.5):
    max_height = 0
    min_height = float('inf')
    all_lines = []
    rects_dict = {}

    for page_idx, page in enumerate(pages, start=1):
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

        data, max_height, min_height = create_clusters(lines, big_threshold, medium_threshold, small_threshold,
                                                       max_height, min_height,
                                                       page_idx)
        all_lines += data
        rects_dict[page_idx] = rects

    sections = split_line_per_section(all_lines=all_lines,
                                      max_height=max_height)

    table_lines = split_section_per_table(sections, max_height)
    title = None

    if len(table_lines) > 0:
        title = get_title(table_lines[0], max_height)

    tables = []
    for lines in table_lines:
        if title is not None:
            descriptions, column_barriers, value_columns, rows = group_vertically(lines, rects_dict,
                                                                                  max_height, min_height)
        else:
            descriptions, column_barriers, value_columns, rows = group_vertically(
                lines, rects_dict, max_height, min_height
            )

        is_nested = check_table_nested(column_barriers, rows)

        try:
            if is_nested:
                final_result = nested_format_final_result(column_barriers, value_columns, rows)
            else:
                final_result = format_final_result(column_barriers, value_columns, rows)
        except ValueError as e:
            print(f'Error formatting final result for title: {title}')
            return None, None

        tables.append({
            'data': final_result,
            'descriptions': descriptions,
            'nested_result': is_nested
        })
        main_page_parsed = True

    return title, tables


def split_section_per_table(sections, max_height: int):
    new_sections = []

    for section in sections:
        split_x = None

        for line in section:
            if len(line) == 4 and line[0]['height'] == max_height:
                second_max_x = line[1]['max_x']
                third_min_x = line[2]['min_x']
                split_x = (second_max_x + third_min_x) / 2
                break

        if split_x is None:
            cleaned_section = [line for line in section if line]
            if cleaned_section:
                new_sections.append(cleaned_section)
            continue

        left_part = []
        right_part = []
        for line in section:
            left_line = [word for word in line if word['max_x'] <= split_x]
            right_line = [word for word in line if word['min_x'] > split_x]

            if not left_line and not right_line:
                left_line = line.copy()

            if left_line:
                left_part.append(left_line)
            if right_line:
                right_part.append(right_line)

        if left_part:
            new_sections.append(left_part)
        if right_part:
            new_sections.append(right_part)

    return new_sections


def split_line_per_section(all_lines, max_height: int):
    split_lines = []
    current_table = []
    seen_content = False

    for line in all_lines:
        if not line:
            continue

        heights = [word["height"] for word in line]
        is_header = any(h == max_height for h in heights)
        is_content = not is_header

        if is_header:
            if seen_content:
                split_lines.append(current_table)
                current_table = []
                seen_content = False
            current_table.append(line)

        elif is_content:
            if current_table:
                seen_content = True
                current_table.append(line)

    if current_table:
        split_lines.append(current_table)

    return split_lines


def create_clusters(lines, big_threshold, medium_threshold, small_threshold, max_height, min_height, page_index):
    clusters_per_line = []

    for y, items in sorted(lines.items()):
        items.sort(key=lambda x: x["x0"])
        clusters = []
        current_cluster = [items[0]]

        for i in range(1, len(items)):
            prev_x = items[i - 1]["x1"]
            curr_x = items[i]["x0"]
            gap = curr_x - prev_x
            gap_threshold = big_threshold

            h = items[i]['height']

            if h < 9:
                gap_threshold = small_threshold
            elif h < 11:
                gap_threshold = medium_threshold

            if gap > gap_threshold:
                top_vals = [w["top"] for w in current_cluster]
                bottom_vals = [w["bottom"] for w in current_cluster]
                cluster_height = round(max(bottom_vals) - min(top_vals))
                max_height = max(max_height, cluster_height)
                min_height = min(min_height, cluster_height)

                clusters.append(make_cluster(current_cluster, height=cluster_height, page_index=page_index))
                current_cluster = [items[i]]
            else:
                current_cluster.append(items[i])

        # process last cluster
        top_vals = [w["top"] for w in current_cluster]
        bottom_vals = [w["bottom"] for w in current_cluster]
        cluster_height = round(max(bottom_vals) - min(top_vals))
        max_height = max(max_height, cluster_height)
        min_height = min(min_height, cluster_height)

        clusters.append(make_cluster(current_cluster, height=cluster_height, page_index=page_index))
        clusters_per_line.append(clusters)

    return clusters_per_line, max_height, min_height


def make_cluster(words, height, page_index):
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
        'max_x': x1,
        'page_index': page_index
    }


def group_vertically(clusters,
                     rects,
                     max_height: int,
                     min_height: int):
    """
    Group clusters vertically so that every table row consists of 2 lines,
    separated by horizontal rectangles.
    """
    row_rects = {}
    col_rects = {}

    for page_index, page_rects in rects.items():
        page_row_rects = [rect for rect in page_rects if rect['height'] < 2]
        page_row_rects.sort(key=lambda r: r['top'])  # sort by vertical position
        page_col_rects = [rect for rect in page_rects if rect['height'] > 10]

        row_rects[page_index] = page_row_rects
        col_rects[page_index] = page_col_rects

    descriptions = []
    value_columns = []
    index = 0

    for index, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue

        if cluster[0]['height'] == max_height:
            descriptions, index = manage_descriptions(clusters, index, max_height)
        elif cluster[0]['height'] == min_height:
            value_columns, index = manage_value_columns(clusters, index, min_height, col_rects)
            break

    clusters = clusters[index:]
    column_barriers = get_description_column_barriers(clusters, value_columns)
    rows = retrieve_data_rows(clusters, row_rects)
    return descriptions, column_barriers, value_columns, rows


def get_title(clusters, max_height):
    title = ''

    for index, cluster in enumerate(clusters):
        if len(cluster) == 1 and cluster[0]['height'] == max_height:
            title += cluster[0]['text']
        else:
            return title

    return None


def retrieve_data_rows(clusters, rects):
    rows = []

    for page_index, page_rects in rects.items():
        for i in range(1, len(page_rects)):
            prev_line = page_rects[i - 1]
            next_line = page_rects[i]

            row_clusters = []
            for cluster in clusters:
                if cluster[0]['page_index'] != page_index:
                    continue

                cluster_top = cluster[0]['top']
                cluster_bottom = cluster[0]['bottom']

                if cluster_bottom > prev_line['bottom'] and cluster_top < next_line['top']:
                    row_clusters += cluster

            if row_clusters:
                rows.append(row_clusters)

    return rows


def manage_descriptions(clusters, starting_index: int, max_height: int) -> Tuple[List[str], int]:
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
                    'max_x': value['max_x'],
                    'page_index': value['page_index']
                }

    for val in values.values():
        page_idx = val['page_index']
        if page_idx not in col_rects:
            continue

        for col in col_rects[page_idx]:
            col_x0, col_x1 = col["x0"], col["x1"]
            if val["min_x"] >= col_x0 - tolerance and val["max_x"] <= col_x1 + tolerance:
                val["min_x"] = col_x0
                val["max_x"] = col_x1
                break

    seen_texts = {}
    for key, val in values.items():
        text = val["text"]
        if text in seen_texts:
            seen_texts[text] += 1
            val["text"] = f"{text}_{seen_texts[text]}"
        else:
            seen_texts[text] = 0

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
            "values": {i['text']: "" for i in value_columns},
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
                    row_data["values"][matched_col["text"]] = (
                            row_data["values"][matched_col["text"]] + " " + word["text"]
                    ).strip()

        results.append(row_data)

    return results


def check_table_nested(column_barriers, rows):
    tolerance = 2
    barrier = column_barriers[0]
    previous_x = None

    for row in rows:
        for word in row:
            center_x = (word["min_x"] + word["max_x"]) / 2
            barrier_min, barrier_max = barrier["barrier"]
            if barrier_min - tolerance <= center_x <= barrier_max + tolerance:
                current_x = word['min_x']
                if previous_x is not None and \
                        round(current_x) != round(previous_x):
                    return True
                previous_x = current_x

    return False


def nested_format_final_result(
        column_barriers: List[Dict],
        value_columns: List[Dict],
        rows: List[List[Dict]],
        tolerance_for_value: int = 10,
        tolerance_for_top: int = 3,
        indent_tolerance: int = 1.5
) -> Dict:
    """
    Build a nested dictionary based on description columns hierarchy and value columns.
    Simplified: alignment_left is always True, only first description word per row is used.
    """

    final_result: Dict = {}
    parent_stack: List[str] = []
    prev_min_x = None

    def get_first_desc_word(row: List[Dict]) -> Dict:
        barrier = column_barriers[0]
        first_desc_word = None
        for w in row:
            center_x = (w["min_x"] + w["max_x"]) / 2
            barrier_min, barrier_max = barrier["barrier"]
            if barrier_min - tolerance_for_top <= center_x <= barrier_max + tolerance_for_top:
                if first_desc_word is None:
                    first_desc_word = w
                else:
                    first_desc_word['text'] += f' {w['text']}'

        return first_desc_word

    def get_row_values(row: List[Dict]) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for idx, val_col in enumerate(value_columns, start=1):
            found_value = ""
            for word in row:
                center_x = (word["min_x"] + word["max_x"]) / 2
                if val_col["min_x"] - tolerance_for_value <= center_x <= val_col["max_x"] + tolerance_for_value:
                    found_value = word["text"]
                    break
            values[val_col['text']] = found_value
        return values

    for row_idx, row in enumerate(rows):
        desc_word = get_first_desc_word(row)
        if desc_word is None:
            raise ValueError(f"Row {row_idx} has no description word.")

        current_min_x = desc_word["min_x"]
        text = desc_word["text"]

        if prev_min_x is not None:
            if current_min_x > prev_min_x + indent_tolerance:
                parent_stack.append(text)
            elif current_min_x < prev_min_x - indent_tolerance:
                if parent_stack:
                    parent_stack.pop()
                    parent_stack.pop()

                parent_stack.append(text)
            else:
                if parent_stack:
                    parent_stack[-1] = text
                else:
                    parent_stack.append(text)
        else:
            parent_stack.append(text)

        current_dict = final_result
        for key in parent_stack[:-1]:
            current_dict = current_dict.setdefault(key, {})

        current_dict[parent_stack[-1]] = get_row_values(row)
        prev_min_x = current_min_x

    return final_result
