# Standard library imports
import os
import re
import base64

# Data manipulation and analysis
import pandas as pd
import numpy as np

# PDF handling and text extraction
import fitz  # PyMuPDF

# Image processing
from PIL import Image

# OCR and visualization

# Machine learning and deep learning

# Networking and web requests
import requests

# Graphs and network analysis
import networkx as nx

# OpenAI API
import openai


# Functions here
# All the functions are defined here


#####  ################################# ################################# RULE BASED BBOX extraction ################################# ################################# ################################# ################################# #################################


# Functions here
# All the functions are defined here


#####  ################################# ################################# RULE BASED BBOX extraction ################################# ################################# ################################# ################################# #################################


def extract_word_positions(pdf_path, page_num):
    page_num = page_num
    doc = fitz.open(pdf_path)
    word_data = []
    page_width = []
    # for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
        for line in block["lines"]:
            wdir = line["dir"]  # writing direction = (cosine, sine)
            if wdir[0] == 0:  # either 90° or 270°
                # print(line['bbox'])
                page.add_redact_annot(line["bbox"])
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    width = page.rect.width
    page_width.append(width)
    words = page.get_text("words")
    for word in words:
        x0, y0, x1, y1, word_text, block_no, line_no, word_no = word
        word_data.append(
            [page_num, x0, y0, x1, y1, word_text, block_no, line_no, word_no]
        )

    df_words = pd.DataFrame(
        word_data,
        columns=[
            "page_num",
            "x0",
            "y0",
            "x1",
            "y1",
            "word_text",
            "block_no",
            "line_no",
            "word_no",
        ],
    )
    return df_words, page_width


"""
def merge_words_into_lines(df_words, y_threshold=5):
    df_words = df_words.sort_values(by=["page_num", "y0", "x0"]).reset_index(drop=True)
    lines = []
    current_line = []
    current_line_num = 0

    for i, row in df_words.iterrows():
        if not current_line:
            current_line.append(row)
        else:
            prev_word = current_line[-1]
            if row["page_num"] == prev_word["page_num"] and abs(row["y0"] - prev_word["y0"]) <= y_threshold:
                current_line.append(row)
            else:
                lines.append((current_line_num, current_line))
                current_line = [row]
                current_line_num += 1

    if current_line:
        lines.append((current_line_num, current_line))

    line_data = []
    for line_num, words in lines:
        for word in words:
            line_data.append([line_num, word["x0"], word["y0"], word["x1"], word["y1"], word["word_text"]])

    df_lines = pd.DataFrame(line_data, columns=["line_num", "x0", "y0", "x1", "y1", "word_text"])
    return df_lines
"""


def merge_words_into_lines(df_words, page_widths, y_threshold=5):
    df_words = df_words.sort_values(by=["page_num", "y0", "x0"]).reset_index(drop=True)
    lines = []
    current_line = []
    current_line_num = 0

    for i, row in df_words.iterrows():
        if not current_line:
            current_line.append(row)
        else:
            prev_word = current_line[-1]
            if (
                row["page_num"] == prev_word["page_num"]
                and abs(row["y0"] - prev_word["y0"]) <= y_threshold
            ):
                current_line.append(row)
            else:
                lines.append((current_line_num, current_line))
                current_line = [row]
                current_line_num += 1

    if current_line:
        lines.append((current_line_num, current_line))

    line_data = []
    flagged_lines = []
    for line_num, words in lines:
        page_num = words[0]["page_num"]
        page_num = (
            0  # Assuming page numbers start from 0 (as we are checking for every page)
        )
        x = page_widths[page_num] / 10
        if all(word["x0"] < 5 * x for word in words):
            flagged_lines.append(line_num)
        for word in words:
            line_data.append(
                [
                    line_num,
                    word["x0"],
                    word["y0"],
                    word["x1"],
                    word["y1"],
                    word["word_text"],
                ]
            )

    df_lines = pd.DataFrame(
        line_data, columns=["line_num", "x0", "y0", "x1", "y1", "word_text"]
    )
    return df_lines, flagged_lines


def find_lines_with_large_x_diff(df_lines, threshold=50):
    lines_with_large_diff = df_lines[df_lines["distance"] > threshold][
        "line_num"
    ].unique()
    return lines_with_large_diff


# Function to calculate distances between words in the same line
def calculate_distances(df):
    distances = []

    # Group by line_num
    grouped = df.groupby("line_num")

    for line_num, group in grouped:
        group = group.sort_values("x0")
        previous_x1 = None

        for index, row in group.iterrows():
            if previous_x1 is not None:
                distance = row["x0"] - previous_x1
                distance = round(distance)
                distances.append(
                    {
                        "line_num": line_num,
                        "word1": previous_word,
                        "word2": row["word_text"],
                        "distance": distance,
                    }
                )
            previous_x1 = row["x1"]
            previous_word = row["word_text"]

    return pd.DataFrame(distances)


def union_of_lists(list1, list2):
    # Convert lists to sets and perform union
    union_set = set(list1) | set(list2)
    # Convert the set back to a list (optional, as sets can be used directly)
    union_list = list(union_set)
    return union_list


def consecutive_sublists(lst, min_length=3):
    lst = sorted(lst)
    sublists = []
    current_sublist = []

    for num in lst:
        if not current_sublist or num == current_sublist[-1] + 1:
            current_sublist.append(num)
        else:
            if len(current_sublist) >= min_length:
                sublists.append(current_sublist)
            current_sublist = [num]

    if len(current_sublist) >= min_length:
        sublists.append(current_sublist)

    return sublists


def check_y_diff_within_threshold_text_classification(df, sublist, threshold=45):
    # Filter the dataframe for the lines in the sublist
    filtered_df = df[df["line_num"].isin(sublist)]

    # Get the first entry per line number
    first_entries_per_line = filtered_df.groupby("line_num").first().reset_index()

    # Extract the y0 values
    y0_values = first_entries_per_line["y0"].values
    # Check differences and remove elements if needed
    while True:
        y_diff = abs(pd.Series(y0_values).diff().dropna())
        y_diff = y_diff.tolist()
        if len(y_diff) > 0:
            y_diff.insert(0, y_diff[0])
        y_diff = pd.Series(y_diff)

        exceed_indices = y_diff[y_diff > threshold].index
        if exceed_indices.empty:
            return True

        # Remove the element causing the exceedance
        exceed_index = exceed_indices[0]  # Adjust for the dropped first value

        y0_values = list(y0_values)

        del y0_values[exceed_index]

        if len(y0_values) <= 2:
            return False  # If there are fewer than 2 elements, we can't compare


def extract_image_blocks(pdf_path, page_num=5):
    # Open the PDF document
    inbuilt_regions = []
    xy_regions = []
    doc = fitz.open(pdf_path)

    # Iterate through all the pages and print image block coordinates

    print(f"Page number: {page_num}")

    # Get the page
    page = doc[page_num]

    # Extract text as dictionary
    d = page.get_text("dict")

    # Get blocks from the dictionary
    blocks = d["blocks"]

    # Filter image blocks
    imgblocks = [b for b in blocks if b["type"] == 1]

    # Print coordinates of image blocks
    if imgblocks:
        for imgblock in imgblocks:
            x0, y0, x1, y1 = imgblock["bbox"]
            if abs(y0 - y1) >= 40:
                print(x0, y0, x1, y1)
                inbuilt_regions.append((y0, y1))
                xy_regions.append((x0, y0, x1, y1))
        return inbuilt_regions, xy_regions
    else:
        print("No image blocks found.")
        return inbuilt_regions, xy_regions


def detect_figure_regions(pdf_path, y_diff_threshold=5, page_num=5):
    # Open the PDF file
    document = fitz.open(pdf_path)

    figure_regions = []

    page_num = page_num
    page = document.load_page(page_num)
    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
        for line in block["lines"]:
            wdir = line["dir"]  # writing direction = (cosine, sine)
            if wdir[0] == 0:  # either 90° or 270°
                # print(line['bbox'])
                page.add_redact_annot(line["bbox"])
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    words = page.get_text("words")

    # Sort words by y0 coordinate
    words.sort(key=lambda w: w[3])

    # Group words into lines based on y_diff_threshold
    lines = []
    current_line = []
    current_y = words[0][3] if words else 0  # Handle case with no words

    for word in words:
        x0, y0, x1, y1, word_str, _, _, _ = word
        if abs(y0 - current_y) <= y_diff_threshold:
            current_line.append(word)
        else:
            if current_line:  # Add non-empty lines only
                lines.append(current_line)
            current_line = [word]
            current_y = y0
    if current_line:  # Add the last line if it's not empty
        lines.append(current_line)

    # Filter out any empty lines that may have been mistakenly added
    lines = [line for line in lines if line]

    # Calculate the midpoints of y-coordinates of lines
    y_midpoints = [
        (min(line, key=lambda w: w[1])[1] + max(line, key=lambda w: w[3])[3]) / 2
        for line in lines
    ]

    # Calculate the differences between consecutive midpoints
    y_diffs = np.diff(y_midpoints)

    # Calculate the median of y differences
    median_y_diff = np.median(y_diffs) if y_diffs.size > 0 else 0

    # Find large gaps in y differences
    large_gaps = np.where(y_diffs > 6 * median_y_diff)[0]

    # Handle case where there's a large gap before the first line of text
    if lines and lines[0][0][1] > 6 * median_y_diff:
        top = 0
        bottom = min(lines[0], key=lambda w: w[1])[1]
        figure_regions.append((top, bottom))

    for gap in large_gaps:
        # Determine the top and bottom coordinates of the figure region
        top = max(lines[gap], key=lambda w: w[3])[3]
        bottom = min(lines[gap + 1], key=lambda w: w[1])[1]
        figure_regions.append((top, bottom))

    # Handle case where there's a large gap after the last line of text
    if (
        lines
        and (page.rect.y1 - max(lines[-1], key=lambda w: w[3])[3]) > 3 * median_y_diff
    ):
        top = max(lines[-1], key=lambda w: w[3])[3]
        bottom = page.rect.y1
        figure_regions.append((top, bottom))

    return figure_regions


def get_pdf_page_dimensions(pdf_path, page_number=5):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Select the specific page
    page = pdf_document.load_page(page_number)

    # Get the dimensions
    width = page.rect.width
    height = page.rect.height

    return width, height


def calculate_overlap(region, range_header, range_footnote):
    header_start, header_end = range_header
    footnote_start, footnote_end = range_footnote
    region_start, region_end = region

    # Calculate overlap with header
    header_overlap = max(
        0, min(region_end, header_end) - max(region_start, header_start)
    )

    # Calculate overlap with footnote
    footnote_overlap = max(
        0, min(region_end, footnote_end) - max(region_start, footnote_start)
    )

    # Total overlap
    total_overlap = max(header_overlap, footnote_overlap)

    # Calculate the length of the figure region
    region_length = region_end - region_start

    # Calculate the percentage of overlap
    overlap_percentage = (total_overlap / region_length) * 100

    return overlap_percentage


def filter_regions(regions, range_header, range_footnote, threshold=50):
    filtered_regions = []

    for region in regions:
        overlap_percentage = calculate_overlap(region, range_header, range_footnote)
        if overlap_percentage <= threshold:
            filtered_regions.append(region)

    return filtered_regions


def process_regions(rule_filtered_regions, new_inbuilt_regions):
    """
    Process and adjust regions based on their lengths and return the final bounding box regions.

    Args:
    rule_filtered_regions (list): List of rule filtered regions.
    new_inbuilt_regions (list): List of new inbuilt regions.

    Returns:
    list: Final bounding box regions after processing.
    """

    # Function to merge intervals with threshold consideration
    def merge_intervals_with_threshold(intervals, threshold=3):
        # Sort intervals based on the first value of each tuple (start of interval)
        intervals.sort(key=lambda x: x[0])

        # List to hold the merged intervals
        merged = []

        # Iterate through the sorted intervals
        for interval in intervals:
            # If the list of merged intervals is empty or the gap between the current interval and the previous one is above the threshold
            if not merged or merged[-1][1] + threshold < interval[0]:
                merged.append(interval)
            else:
                # There is an effective overlap, merge the current interval with the previous one
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

        return merged

    # Function to adjust intervals before merging
    def adjust_intervals(intervals):
        adjusted = []
        for start, end in intervals:
            # Add 10 to the start, subtract 10 from the end
            new_start = start + 10
            new_end = end - 10
            adjusted.append((new_start, new_end))
        return adjusted

    if len(rule_filtered_regions) == len(new_inbuilt_regions):
        bbox_regions = new_inbuilt_regions

    elif len(rule_filtered_regions) < len(new_inbuilt_regions):
        new_inbuilt_regions = merge_intervals_with_threshold(new_inbuilt_regions, 3)
        bbox_regions = new_inbuilt_regions

    else:  # len(rule_filtered_regions) > len(new_inbuilt_regions)
        rule_filtered_regions = merge_intervals_with_threshold(rule_filtered_regions, 3)
        rule_filtered_regions = adjust_intervals(rule_filtered_regions)
        bbox_regions = rule_filtered_regions

    return bbox_regions


def get_y0(df, sublist):
    # Filter the dataframe for the lines in the sublist
    filtered_df = df[df["line_num"].isin(sublist)]

    # Get the first entry per line number
    first_entries_per_line = filtered_df.groupby("line_num").first().reset_index()
    # print(first_entries_per_line)

    # Extract the y0 values
    y0_values = first_entries_per_line["y0"].values
    y1_values = first_entries_per_line["y1"].values
    return (y0_values[0], y1_values[-1])


################################################CAPTION MODULE###############################################################


# Function to extract text and figure captions from PDF
def extract_text_and_captions(pdf_path):
    # Enhanced regex pattern for figure captions
    regex_pattern = r'\b(?:fig\.?|figure|panel|measure|analysis|sfig|sfig\.?)\s+[A-H\d]+(?:\.\d+)*(?:\.\d+)?\s*[:\-—.|()]*\s*(?:\|\s*\([a-z]\)\s*)?["\'A-Za-z0-9].*'
    document = fitz.open(pdf_path)
    text_data = {}
    captions = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        text_data[page_num] = text

        # Refined regex search
        page_captions = re.findall(regex_pattern, text, re.IGNORECASE)

        # Filter out false positives
        filtered_captions = []
        for cap in page_captions:
            # Avoid captions that are too short or span multiple lines with unrelated content
            if (
                len(cap.split()) >= 3
                and not re.search(r"\b(?:Annex|Appendix)\b|\nNOTE:", cap)
                and cap.strip()[0].isupper()
            ):
                filtered_captions.append(cap.strip())

        if filtered_captions:
            captions.extend([(page_num, cap) for cap in filtered_captions])

    return text_data, captions


def are_brackets_balanced(s):
    stack = []
    for ch in s:
        if ch in ("(", "{", "["):
            stack.append(ch)
        elif ch in (")", "}", "]"):
            if stack and (
                (stack[-1] == "(" and ch == ")")
                or (stack[-1] == "{" and ch == "}")
                or (stack[-1] == "[" and ch == "]")
            ):
                stack.pop()
            else:
                return False
    return not stack


def is_valid_caption(caption):
    # Filtration 1: Check for long sequences of dots
    if re.search(r"\.{6,}", caption):
        return "No"

    # Filtration 2: Presence of ). at the very start
    if re.search(r"^\d+\.\d+\)\.", caption):
        return "No"

    # Filtration 3: Presence of , after the mention (let's say 23.1 there is a comma after)
    if re.search(r"\d+\.\d+,", caption):
        return "No"

    # Additional Filtration: Check if the first 15 characters have balanced brackets
    if not are_brackets_balanced(caption[:15]):
        return "No"

    # Additional Filtration: Check if there is any numeric character in the string
    if not any(char.isdigit() for char in caption):
        return "No"

    # Additional Filtration: Check if there are at least two uppercase characters
    if sum(1 for char in caption[:20] if char.isupper()) < 2:
        return "No"

    # Additional Filtration: Check if in the first 15 characters there is a comma after a number
    if re.search(r"\d,", caption[:15]):
        return "No"

    # Additional Filtration: Check if in the first 15 characters there is a ; after a number
    if re.search(r"\d;", caption[:15]):
        return "No"

    return "Yes"


def complete_captions(df, pdf_path):
    """
    Complete half-filled captions in the DataFrame using text from the PDF, by expanding the search area below the detected caption line.

    Args:
    df (pandas.DataFrame): DataFrame containing the captions.
    pdf_path (str): Path to the PDF file.

    Returns:
    pandas.DataFrame: DataFrame with potentially extended captions.
    """
    try:
        doc = fitz.open(pdf_path)  # Attempt to open the PDF file
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return df  # Return the original DataFrame on failure

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        page_number = row["Page"]
        caption_start = row["Caption"].strip()  # Trim whitespace

        try:
            page = doc.load_page(page_number)  # PyMuPDF uses 0-based indexing
        except Exception as e:
            print(f"Error loading page {page_number}: {e}")
            continue

        # Search for the first instance of the caption text to get its bounding box
        caption_start = caption_start[: round(len(caption_start) / 2)]
        text_instances = page.search_for(caption_start)
        if text_instances:
            first_instance = text_instances[
                0
            ]  # Assume the first instance is what we want
            bbox_height = (
                first_instance.y1 - first_instance.y0
            )  # Calculate height of the bbox
            additional_area = (
                first_instance.x0 - 50,
                first_instance.y1,
                first_instance.x1 + 50,
                first_instance.y1 + (2.0 * bbox_height),
            )

            bbox = text_instances[0]  # Use the first instance's bbox
            # Extend the y1 coordinate by 20 points
            extended_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1 + 20)
            df.at[index, "BBox"] = extended_bbox  # Store the extended bounding box
            y_mid = (extended_bbox.y0 + extended_bbox.y1) / 2
            x_mid = (extended_bbox.x0 + extended_bbox.x1) / 2
            df.at[index, "y_mid"] = round(y_mid, 2)
            df.at[index, "x_mid"] = round(x_mid, 2)

            # Search in the expanded area below the original caption
            additional_text = page.get_textbox(additional_area)
            if additional_text:
                # Combine the original caption with the newly found text
                full_caption = caption_start + " " + additional_text.strip()
                df.at[index, "Caption"] = full_caption

    return df


def find_nearest_captions(df_captions, figure_bboxes):
    """
    Matches each figure to the nearest caption based on the minimum vertical distance.

    Args:
    df_captions (pandas.DataFrame): DataFrame containing 'Caption' and 'y_mid' columns.
    figure_bboxes (list of dicts): List of dictionaries, each containing 'id', 'y_start', and 'y_end' keys.

    Returns:
    dict: A dictionary mapping each figure ID to the caption that is closest based on vertical distance.
    """
    # Create a bipartite graph
    G = nx.Graph()

    # Adding nodes with two distinct sets: figures and captions
    for figure in figure_bboxes:
        G.add_node(figure["id"], bipartite=0)  # Figure nodes
    for index, row in df_captions.iterrows():
        G.add_node(row["Caption"], bipartite=1)  # Caption nodes

    # Adding edges based on vertical distance
    for figure in figure_bboxes:
        for index, row in df_captions.iterrows():
            # Calculate minimum distance between the mid-point of the caption and the start or end of the figure bbox
            distance = min(
                abs(figure["y_start"] - row["y_mid"]),
                abs(figure["y_end"] - row["y_mid"]),
            )
            G.add_edge(figure["id"], row["Caption"], weight=distance)

    # Compute the minimum weight full matching on the bipartite graph
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
        G, top_nodes=[f["id"] for f in figure_bboxes]
    )

    return matching


################################################  FIGURE CLASSIFICATION   ###############################################################


class ImageClassifier:
    def __init__(self, api_key, image_path):
        self.api_key = api_key
        self.image_path = image_path
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def encode_image(self):
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def classify_image(self):
        base64_image = self.encode_image()

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Check if this figure represents a forest plot or not. Simply return the class of the image. Here are the classes: Forest Plot, Not Forest Plot. You can only return one class from that list.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        return response.json()


########################################## PDF TO IMAGE OR VICE VERSA UTILITY FUNCTIONS *************************************************************************


def pdf_to_image_coordinates(
    pdf_x, pdf_y, pdf_width, pdf_height, img_width, img_height
):

    # Calculate scaling factors
    x_scale = img_width / pdf_width
    y_scale = img_height / pdf_height

    # Convert PDF coordinates to image coordinates
    img_x = pdf_x * x_scale
    img_y = pdf_y * y_scale

    return (img_x, img_y)


def get_pdf_dimensions(pdf_path):

    # Open the PDF file
    doc = fitz.open(pdf_path)
    # Get the first page
    page = doc[0]
    # Extract the dimensions
    width, height = page.rect.width, page.rect.height
    # Close the document
    doc.close()
    return (width, height)


def get_image_dimensions(image_path):

    # Open the image
    with Image.open(image_path) as img:
        # Extract dimensions
        width, height = img.size
    return (width, height)


def crop_image(image_path, x0, y0, x1, y1, save_folder, mention=["TEST"]):
    """
    Crops an image to the specified coordinates and saves the cropped image to a specified folder.

    Args:
    image_path (str): The file path to the source image.
    x0 (int): The left boundary of the rectangle to be cropped.
    y0 (int): The upper boundary of the rectangle to be cropped.
    x1 (int): The right boundary of the rectangle to be cropped.
    y1 (int): The lower boundary of the rectangle to be cropped.
    save_folder (str): The folder path where the cropped image will be saved.

    Returns:
    str: Path of the saved cropped image.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Open the image
    with Image.open(image_path) as img:
        # Crop the image
        cropped_img = img.crop((x0, y0, x1, y1))

        # Prepare the path for saving the cropped image

        save_path = os.path.join(save_folder, (mention[0] + ".png"))

        # Save the cropped image
        cropped_img.save(save_path)

    return save_path


def save_pdf_pages_as_images(pdf_path, page_numbers, output_folder):
    doc = fitz.open(pdf_path)

    # Loop through the specified page numbers
    for page_number in page_numbers:
        print(page_number)
        # Ensure the page number is within the range of available pages
        if page_number < 0 or page_number >= len(doc):
            print(f"Page number {page_number} is out of range.")
            continue
        # Load the page
        page = doc.load_page(page_number)
        # Get the pixmap of the page
        pixmap = page.get_pixmap(dpi=300)
        # Save the pixmap as an image
        output_path = f"{output_folder}/page_{page_number}.png"
        pixmap.save(output_path)

    # Close the PDF document
    doc.close()


# Example usage:
# save_pdf_pages_as_images('example.pdf', [0, 1, 2], 'output_folder')


# Example usage:
# save_pdf_pages_as_images('example.pdf', [0, 1, 2], 'output_folder')


######################################################## OCR PIPELINE  #####################################################################


def process_ocr_results(ocr_results, x_range=None, y_range=None):
    """
    Processes OCR results to extract bounding box coordinates and text, and optionally filters them.

    :param ocr_results: The OCR results in a specific format.
    :param x_range: Optional tuple of (min_x, max_x) to filter results based on x-coordinate.
    :param y_range: Optional tuple of (min_y, max_y) to filter results based on y-coordinate.
    :return: A pandas DataFrame with columns for bounding box coordinates (X0, X1, Y0, Y1) and detected text.
    """
    bbox, text_blocks = [], []

    for item in ocr_results[0]:
        a_point_x, a_point_y = item[0][0][0], item[0][0][1]
        c_point_x, c_point_y = item[0][2][0], item[0][2][1]
        avg_x, avg_y = (a_point_x + c_point_x) / 2, (a_point_y + c_point_y) / 2

        # Uncomment and modify the conditional below if filtering is needed
        # if (x_range is None or x_range[0] < avg_x < x_range[1]) and (y_range is None or y_range[0] < avg_y < y_range[1]):
        bbox.append(item[0])
        text_blocks.append(item[1][0])

    y0ss = [sublist[0][1] for sublist in bbox]
    integer_list = [int(l) for l in y0ss]

    # Prepare the DataFrame
    df = pd.DataFrame(
        {
            "X0": [sublist[0][0] for sublist in bbox],
            "X1": [sublist[2][0] for sublist in bbox],
            "Y0": [sublist[0][1] for sublist in bbox],
            "Y1": [sublist[2][1] for sublist in bbox],
            "Text": text_blocks,
        }
    )

    return df, integer_list


def find_unique_lines_row(df):
    # Calculate heights and their mean
    heights = [row["Y1"] - row["Y0"] for i, row in df.iterrows()]
    height_mean = sum(heights) / len(heights)

    # Initialize list to keep track of unique lines
    h_lines = []
    sensitivity = height_mean / 2

    # Identify unique lines based on sensitivity
    for y0 in df["Y0"]:
        found = False
        for c in h_lines:
            if abs(y0 - c) < sensitivity:
                found = True
                break
        if not found:
            h_lines.append(y0)

    return h_lines


def find_unique_lines_column(df):
    # Calculate heights and their mean
    heights = [row["X1"] - row["X0"] for i, row in df.iterrows()]
    height_mean = sum(heights) / len(heights)

    # Initialize list to keep track of unique lines
    h_lines = []
    sensitivity = height_mean / 2

    # Identify unique lines based on sensitivity
    for y0 in df["X0"]:
        found = False
        for c in h_lines:
            if abs(y0 - c) < sensitivity:
                found = True
                break
        if not found:
            h_lines.append(y0)

    return h_lines


def find_segment(midpoint, lines):
    """Helper function to determine the segment number based on the midpoint."""
    for i, line in enumerate(lines):
        if midpoint < line:
            return i
    return len(lines)  # for the last segment


def assign_row_col_numbers(df, row_lines, column_lines):
    # Calculate midpoints
    df["MidX"] = (df["X0"] + df["X1"]) / 2
    df["MidY"] = (df["Y0"] + df["Y1"]) / 2

    # Determine row and column numbers
    df["RowNumber"] = df["MidY"].apply(lambda y: find_segment(y, row_lines))
    df["ColumnNumber"] = df["MidX"].apply(lambda x: find_segment(x, column_lines))
    return df


def process_image_to_latex(
    saved_image_path,
    ocr,
    process_ocr_results,
    find_unique_lines_row,
    find_unique_lines_column,
    assign_row_col_numbers,
):
    # Perform OCR on the image
    result = ocr.ocr(saved_image_path, cls=True)
    print(result)

    # Process OCR results
    df_new, integer_list = process_ocr_results(result, x_range=None, y_range=None)
    print(df_new)
    print(integer_list)

    # Find and sort unique row lines
    row_lines = find_unique_lines_row(df_new)
    row_lines.sort()

    # Find and sort unique column lines
    column_lines = find_unique_lines_column(df_new)
    column_lines.sort()

    # Assign row and column numbers to the DataFrame
    updated_df = assign_row_col_numbers(df_new, row_lines, column_lines)

    # Group by RowNumber and ColumnNumber and concatenate texts
    grouped = (
        df_new.groupby(["RowNumber", "ColumnNumber"])["Text"]
        .agg(" ".join)
        .reset_index()
    )

    # Pivot the DataFrame to create a mapped view
    mapped_df = grouped.pivot(index="RowNumber", columns="ColumnNumber", values="Text")

    # Fill NaN with empty strings if needed
    mapped_df = mapped_df.fillna("")
    print(mapped_df)

    # Convert the DataFrame to a LaTeX table
    latex_table = mapped_df.to_latex(index=True, header=True)
    print(latex_table)
    return latex_table


# Function to classify table headers from a LaTeX table using GPT-4
def classify_headers_from_latex_with_gpt(latex_content, caption_text):
    # Preprocess the LaTeX content to get only the table part
    table_pattern = re.compile(r"\\begin{tabular}.*?\\end{tabular}", re.DOTALL)
    print("abcd")
    table_match = table_pattern.search(latex_content)
    print("efgh")

    # If no table found, raise an error
    if not table_match:
        raise ValueError("No table found in the LaTeX content")

    table_content = latex_content
    # Escape curly braces in the LaTeX content
    escaped_table_content = table_content.replace("{", "{{").replace("}", "}}")
    print("ijkl")

    # Prepare the prompt for GPT-4
    prompt = f"""
    Describe the numeric cells in the provided latex table using the JSON templates below. Proceed row by row, from left to right and top to bottom. Output one JSON description per line. Use the appropriate template based on the type of numeric cell: "Effect Size". For any unanswerable attributes in the templates, set their value to the placeholder "xx" or "yy" if it is a string type. Cells that do not describe effect size please ignore them and you can also use caption text to extract interventions and outcomes if needed as you are en expert in this. Also, ignore overall/total effect sizes.

    JSON Templates:

    {{"value": "xx", "type": "Effect Size", "confidence interval": ["xx", "yy"], "author": "xx", "year": "xx", "country": "xx", "intervention": "xx", "outcome": "xx", "sample size": "xx", "programme": "xx", "group": "xx"}}

    caption_text:
    {caption_text}

    Latex:
    {escaped_table_content}

    Please provide the results in JSON format.

    Json:
    """

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts and maps details from latex table related to economic RCTs/impact evaluations.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0,
    )
    print(response)
    print("mnop")

    # Extract the classifications from the response
    classifications = response["choices"][0]["message"]["content"].strip()
    print("qrst")
    return classifications


################## POST PROCESSING FUNCTIONS ########################################


def adjust_signs_to_match_rule(x, y, z):
    def custom_round_plus(value, decimals):
        # Add a small bias before rounding to ensure correct rounding behavior
        bias = 0.5 * 10 ** (-decimals)
        return round(value + bias, decimals)

    def custom_round_minus(value, decimals):
        # Add a small bias before rounding to ensure correct rounding behavior
        bias = 0.5 * 10 ** (-decimals)
        return round(value - bias, decimals)

    # List of possible sign adjustments
    adjustments = [(y, z), (-y, z), (y, -z), (-y, -z)]

    # Check each adjustment for both x and -x
    for adjusted_y, adjusted_z in adjustments:
        if (
            round((adjusted_y + adjusted_z) / 2, 2) == x
            or custom_round_plus((adjusted_y + adjusted_z) / 2, 2) == x
            or custom_round_minus((adjusted_y + adjusted_z) / 2, 2) == x
        ):
            return x, adjusted_y, adjusted_z
        if (
            round((adjusted_y + adjusted_z) / 2, 2) == -x
            or custom_round_plus((adjusted_y + adjusted_z) / 2, 2) == -x
            or custom_round_minus((adjusted_y + adjusted_z) / 2, 2) == -x
        ):
            return -x, adjusted_y, adjusted_z

    return None


def create_output_folder(output_folder_path, output_folder_name):
    # Get the root directory
    # root_directory = os.getcwd()  # This gets the current working directory

    # Concatenate the root directory with the output folder name
    output_folder = os.path.join(output_folder_path, output_folder_name)

    # Check if the output folder exists
    if not os.path.exists(output_folder):
        # If it doesn't exist, create the folder
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created successfully.")
    else:
        print(f"Output folder '{output_folder}' already exists.")
    return output_folder
