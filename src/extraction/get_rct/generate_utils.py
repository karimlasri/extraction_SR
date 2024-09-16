import json
import time
import fitz
import re
import ast
import pandas as pd
from config.keywords_config import KEYWORDS

from utils import (
    outputs_to_objects,
    get_objects,
    get_image,
    get_image_size,
    get_transform,
    objects_to_crops,
    oriented_process_cropped_table_image,
    process_ocr_results,
    classify_headers_from_latex_with_gpt,
    assign_row_col_numbers,
    process_and_save_cropped_image,
    find_unique_lines_column,
    find_unique_lines_row,
)


def load_document(path_of_pdf):
    """Load the document using PyMuPDF."""
    return fitz.open(path_of_pdf)


def get_page_info(doc, page_num):
    """Extracts information from the page such as text, bounding boxes, and rotation."""
    page = doc.load_page(page_num)
    page_text = page.get_text()
    bbox_log = page.get_bboxlog()
    page_rotation = page.rotation
    return page, page_text, bbox_log, page_rotation


def detect_objects(image, model, detection_transform, device):
    """Detect objects in the image using the specified model and transformation."""
    return get_objects(image, model, detection_transform, device)


def should_process_page(
    page_mix_ocr, page_rotation, page_searchable, page_table_rotated
):
    """Check if the page should be processed based on OCR, rotation, searchability, and table orientation."""
    return (
        page_mix_ocr or page_rotation != 0 or not page_searchable or page_table_rotated
    )


def process_table_image(cropped_table, structure_transform, structure_model, device):
    """Process a cropped table image using a structure model to detect table elements."""
    return oriented_process_cropped_table_image(
        cropped_table, device, structure_transform, structure_model, outputs_to_objects
    )


def extract_text_from_ocr(ocr_img_path, ocr):
    """Extract text from an image using OCR."""
    result = ocr.ocr(ocr_img_path, cls=True)
    df, integer_list = process_ocr_results(result, x_range=None, y_range=None)
    return df, " ".join(df["Text"].values)


def find_keywords(text_string, keywords):
    """Search for keywords within the given text string."""
    text_string_lower = text_string.lower()
    keywords_lower = [keyword.lower() for keyword in keywords]
    return any(keyword in text_string_lower for keyword in keywords_lower)


def process_crops(
    crops, device, structure_transform, structure_model, ocr, temp_output_folder
):
    """Processes each crop and extracts table details."""
    cells = []
    for i, crop in enumerate(crops):
        cropped_table = crop["image"]
        process_table_images, rgb_cropped_image = process_table_image(
            cropped_table, structure_transform, structure_model, device
        )
        cells.append(process_table_images)
        ocr_img_path = process_and_save_cropped_image(
            rgb_cropped_image, temp_output_folder
        )
        df, text_string = extract_text_from_ocr(ocr_img_path, ocr)
        yield df, text_string


def classify_and_extract(df, page_num):
    """Classify headers using GPT and extract JSON-like objects."""
    row_lines = find_unique_lines_row(df)
    column_lines = find_unique_lines_column(df)
    row_lines.sort()
    column_lines.sort()
    updated_df = assign_row_col_numbers(df, row_lines, column_lines)

    grouped = (
        updated_df.groupby(["RowNumber", "ColumnNumber"])["Text"]
        .agg(" ".join)
        .reset_index()
    )
    mapped_df = grouped.pivot(
        index="RowNumber", columns="ColumnNumber", values="Text"
    ).fillna("")

    latex_table = mapped_df.to_latex(index=True, header=True)
    classifications = classify_headers_from_latex_with_gpt(latex_table)

    # Extract JSON-like objects
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, classifications, re.DOTALL)
    total = []
    for match in matches:
        try:
            json_data = json.loads(match)
            total.append(json_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
    return total


def main(
    path_of_pdf,
    Pages_Contains_Tables,
    detection_class_thresholds,
    output_folder_image_path,
    model,
    structure_model,
    device,
    crop_padding,
    ocr,
    temp_output_folder,
):
    """Main function to process the PDF and extract table data."""
    doc = load_document(path_of_pdf)
    Dictionary = {}
    time.time()
    list_of_pages = []

    for page_num in Pages_Contains_Tables:
        file_path = output_folder_image_path + f"/page_{page_num}.png"
        page, page_text, bbox_log, page_rotation = get_page_info(doc, page_num)

        image = get_image(file_path)
        img_width, img_height = get_image_size(file_path)

        detection_transform, structure_transform = get_transform()
        objects = detect_objects(image, model, detection_transform, device)

        page_mix_ocr = any("ignore-text" in bbox[0] for bbox in bbox_log)
        page_searchable = page_text.strip()
        page_table_rotated = (
            objects[0]["label"] == "table rotated" if objects else False
        )

        if should_process_page(
            page_mix_ocr, page_rotation, page_searchable, page_table_rotated
        ):
            try:
                tables_crops = objects_to_crops(
                    image, [], objects, detection_class_thresholds, padding=crop_padding
                )
                for df, text_string in process_crops(
                    tables_crops,
                    device,
                    structure_transform,
                    structure_model,
                    ocr,
                    temp_output_folder,
                ):
                    if find_keywords(text_string, KEYWORDS):
                        list_of_pages.append(page_num)
                        total = classify_and_extract(df, page_num)
                        Dictionary[f"page_{page_num + 1}"] = total
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

        elif objects and objects[0]["label"] == "table":
            try:
                tables_crops = objects_to_crops(
                    image, [], objects, detection_class_thresholds, padding=crop_padding
                )
                for df, text_string in process_crops(
                    tables_crops,
                    device,
                    structure_transform,
                    structure_model,
                    ocr,
                    temp_output_folder,
                ):
                    if find_keywords(text_string, KEYWORDS):
                        list_of_pages.append(page_num)
                        total = classify_and_extract(df, page_num)
                        Dictionary[f"page_{page_num + 1}"] = total
            except Exception as e:
                print(f"Error processing table on page {page_num}: {e}")

        else:
            print("no_table")

    print("Completed processing all pages.")
    return Dictionary


def copy_and_filter_data(dictionary):
    """Copy the dictionary to DataFrame and apply initial filters."""
    dfs = []
    for figure, records in dictionary.items():
        if records:  # Check if records are not empty
            df = pd.DataFrame(records)
            df["Page"] = figure
            dfs.append(df)
    if dfs:  # Check if dfs is not empty before concatenating
        full_df = pd.concat(dfs, ignore_index=True)
        # Apply filtering conditions as needed
        df = full_df[~((full_df["year"] == "xx"))]
        df = df[~((df["author name"] == "xx"))]
        return df
    else:
        return pd.DataFrame(
            columns=["Page", "author name", "type", "year"]
        )  # Return an empty DataFrame with the required columns


def safe_eval_and_clean(text):
    """Safely evaluate strings and clean by removing 'et al.'."""
    if isinstance(text, str):
        try:
            if text.startswith("[") and text.endswith("]"):
                cleaned_text = re.sub(r"et al\.", "", text)
                return ast.literal_eval(cleaned_text)
            else:
                return [re.sub(r"et al\.", "", text).strip()]
        except (ValueError, SyntaxError):
            return [text.strip("[]'")]
    elif isinstance(text, list):
        return [re.sub(r"et al\.", "", author).strip() for author in text]
    else:
        return [text]


def normalize_data(df):
    """Normalize the data by expanding author names into individual rows."""
    normalized_data = []
    for index, row in df.iterrows():
        authors = safe_eval_and_clean(row["author name"])
        for author in authors:
            normalized_data.append(
                [row["Page"], author.strip(), row["type"], row["year"]]
            )
    normalized_df = pd.DataFrame(
        normalized_data, columns=["Page", "author name", "type", "year"]
    )
    normalized_df = normalized_df.dropna(subset=["author name"])
    normalized_df = normalized_df[normalized_df["author name"] != ""]
    return normalized_df


def split_authors(row):
    """Split author names into individual records."""
    authors = row["author name"].replace(" and ", ", ").split(", ")
    return [(author, row["type"], row["year"], row["Page"]) for author in authors]


def expand_authors(df):
    """Apply the split_authors function to expand the DataFrame."""
    expanded_data = df.apply(split_authors, axis=1).explode().tolist()
    expanded_df = pd.DataFrame(
        expanded_data, columns=["author name", "type", "year", "Page"]
    )
    return expanded_df


def remove_conflicts_and_duplicates(df):
    """Remove duplicates and handle conflicts by keeping only RCT records."""
    df = df.drop_duplicates(subset=["author name", "type", "year"])
    df = df.sort_values(
        by=["author name", "year", "type"], ascending=[True, True, False]
    )
    df = df.drop_duplicates(subset=["author name", "year"], keep="first")
    return df


def process_dictionary_to_csv(dictionary):
    """Process the dictionary and return the formatted DataFrame."""
    df = copy_and_filter_data(dictionary)
    if df.empty:
        return df  # Return empty DataFrame if no valid data
    normalized_df = normalize_data(df)
    expanded_df = expand_authors(normalized_df)
    final_df = remove_conflicts_and_duplicates(expanded_df)
    return final_df
