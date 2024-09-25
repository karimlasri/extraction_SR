import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import fitz
from fitz import Rect
from PIL import Image
from torchvision import transforms
import cv2
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
import torch
import math
import logging

# import keywords
from config.keywords_config import char_end_strings, word_end_strings


def initialize_models():
    """
    Initializes and sets up the object detection and structure recognition models,
    loading them onto the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        tuple: A tuple containing the initialized object detection model and the structure recognition model.
    """
    # Set device to CUDA if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the object detection model and move to the appropriate device
    detection_model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    detection_model.to(device)

    # Initialize the structure recognition model and move to the appropriate device
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )
    structure_model.to(device)

    return detection_model, structure_model


def detect_tables_in_document(doc):
    """
    Analyzes each page in the given document to detect which pages likely contain tables.

    Args:
        doc: The document object to analyze.

    Returns:
        list: A list of page numbers that likely contain tables.
    """
    pages_tables_contain = []

    # Iterate through each page in the document
    for page_num, page in enumerate(doc):
        # Extract text blocks and coordinates
        text_blocks, coords = extract_blocks_with_coords(page)

        # Process coordinates to find near duplicates and clean lines
        y_coords = [coord[1] for coord in coords]
        find_near_duplicates(y_coords)
        cleaned_lines = remove_y_duplicates(
            [(coord[:2], coord[2:]) for coord in coords]
        )

        # Count lines with significant length or height
        count_lines = len(
            [
                line
                for line in cleaned_lines
                if abs(line[0][0] - line[1][0]) > 120
                or abs(line[0][1] - line[1][1]) > 120
            ]
        )

        # Analyze text direction for table detection
        count_vertical, count_horizontal = 0, 0
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if line["dir"][0] == 0:
                        count_vertical += 1
                    else:
                        count_horizontal += 1

        # Calculate percentages of vertical and horizontal text lines
        total_lines = count_vertical + count_horizontal + 0.01  # Avoid division by zero
        perc_vertical = count_vertical / total_lines
        perc_horizontal = count_horizontal / total_lines

        # Detect if the page likely contains a table
        contains_table = detect_table(
            page, text_blocks, count_lines, perc_vertical, perc_horizontal
        )
        if contains_table:
            pages_tables_contain.append(page_num)

    return pages_tables_contain


def get_image(file_path):
    """
    Opens an image from the specified file path, converts it to RGB format,
    resizes it to 60% of its original dimensions, and returns the resized image.
    """
    image = Image.open(file_path).convert("RGB")
    width, height = image.size
    image.resize((int(0.6 * width), (int(0.6 * height))))
    return image


def get_image_size(file_path):
    """
    Opens an image from the specified file path, converts it to RGB format,
    and returns the dimensions (width, height) of the image.
    """
    image = Image.open(file_path).convert("RGB")
    return image.size


def get_transform():
    """
    Creates and returns two transformations: one for detection and one for structure.

    Detection transform resizes the image to a maximum size of 800 pixels,
    converts it to a tensor, and normalizes it.
    Structure transform resizes the image to a maximum size of 1000 pixels,
    converts it to a tensor, and applies the same normalization.

    Returns:
        tuple: A tuple containing two transformation pipelines:
               detection_transform and structure_transform.
    """
    detection_transform = transforms.Compose(
        [
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    structure_transform = transforms.Compose(
        [
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return detection_transform, structure_transform


def get_objects(image, model, detection_transform, device):
    """
    Runs object detection on an image using the specified model and transformation.

    Args:
        image (PIL.Image): The input image to detect objects in.
        model (torch.nn.Module): The pre-trained object detection model to be used.
        detection_transform (torchvision.transforms.Compose): The transformation applied to the image before detection.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        list: A list of detected objects with their labels and positions.
    """
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    with torch.no_grad():
        outputs = model(pixel_values)
    objects = outputs_to_objects(outputs, image.size, id2label)
    return objects


class MaxResize(object):
    """
    A class for resizing an image so that its largest dimension (width or height)
    does not exceed a specified maximum size while maintaining the aspect ratio.

    Args:
        max_size (int): The maximum size (in pixels) for the largest dimension
                        of the image. Default is 800.
    """

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


def outputs_to_objects(outputs, img_size, id2label):
    """
    Processes the model outputs to extract detected objects along with their
    labels, confidence scores, and bounding boxes.

    Args:
        outputs (torch.Tensor): The output of the model, containing logits and predicted boxes.
        img_size (tuple): The size of the original input image (width, height).
        id2label (dict): A dictionary mapping label IDs to human-readable labels.

    Returns:
        list: A list of dictionaries, each representing a detected object with
              keys for 'label', 'score', and 'bbox' (bounding box coordinates).
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


def box_cxcywh_to_xyxy(x):
    """
    Converts bounding boxes from center format (cx, cy, w, h) to corner format (x_min, y_min, x_max, y_max).

    Args:
        x (torch.Tensor): A tensor of shape (N, 4), where each box is represented by
                          the center coordinates (cx, cy), width (w), and height (h).

    Returns:
        torch.Tensor: A tensor of shape (N, 4), where each box is represented by
                      the corner coordinates (x_min, y_min, x_max, y_max).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """
    Rescales the bounding boxes from normalized coordinates to absolute coordinates
    based on the size of the original image.

    Args:
        out_bbox (torch.Tensor): A tensor of bounding boxes in (center_x, center_y, width, height) format,
                                 normalized (values between 0 and 1).
        size (tuple): The size of the original image (width, height) as a tuple.

    Returns:
        torch.Tensor: A tensor containing rescaled bounding boxes in (x_min, y_min, x_max, y_max) format,
                      with absolute coordinates relative to the original image size.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def objects_to_crops(img, tokens, objects, class_thresholds, padding=0):
    """
    Processes the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens, while applying class thresholds
    and optional padding.

    Args:
        img (PIL.Image.Image): The input image from which tables are to be cropped.
        tokens (list): A list of token dictionaries, where each token contains a 'bbox' (bounding box).
        objects (list): A list of detected objects, each containing 'label', 'score', and 'bbox'.
        class_thresholds (dict): A dictionary containing confidence thresholds for each object class.
        padding (int, optional): Optional padding to be added around the bounding boxes. Default is 0.

    Returns:
        list: A list of cropped tables, each containing an 'image' and corresponding 'tokens'.
    """

    table_crops = []
    for obj in objects:
        if obj["score"] < class_thresholds[obj["label"]]:
            continue

        cropped_table = {}

        bbox = obj["bbox"]
        bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token["bbox"], bbox) >= 0.5]

        for token in table_tokens:
            token["bbox"] = [
                token["bbox"][0] - bbox[0],
                token["bbox"][1] - bbox[1],
                token["bbox"][2] - bbox[0],
                token["bbox"][3] - bbox[1],
            ]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token["bbox"]
                bbox = [
                    cropped_img.size[0] - bbox[3] - 1,
                    bbox[0],
                    cropped_img.size[0] - bbox[1] - 1,
                    bbox[2],
                ]
                token["bbox"] = bbox
        cropped_table["image"] = cropped_img
        cropped_table["tokens"] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def oriented_process_cropped_table_image(
    cropped_table_image,
    device,
    structure_transform,
    structure_model,
    outputs_to_objects,
):
    """
    Processes a cropped table image to detect cells and their labels using a specified model.

    Parameters:
    - cropped_table_image: A PIL Image object of a cropped table.
    - device: The computation device ('cuda' or 'cpu').
    - structure_transform: A transformation function to apply to the input image before model prediction.
    - structure_model: The pre-loaded PyTorch model for table structure prediction.
    - outputs_to_objects: A function to convert the model's output into a list of detected cells with labels.

    Returns:
    - A list of detected cells and their labels after processing the image through the model.
    """
    # Ensure the image is in RGB
    rgb_image = cropped_table_image.convert("RGB")

    # Apply the transformation and move the tensor to the specified device
    pixel_values = structure_transform(rgb_image).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    # Predict the structure with the model
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # Safely update the id2label dictionary to include "no object"
    structure_id2label = {
        **structure_model.config.id2label,
        len(structure_model.config.id2label): "no object",
    }

    # Convert the model's outputs to objects (cells) with labels
    cells = outputs_to_objects(outputs, rgb_image.size, structure_id2label)

    return cells, rgb_image


def process_and_save_cropped_image(rgb_cropped_image, data_cropped_path):
    """
    Saves a cropped image, reads it back using OpenCV, applies a binary threshold, and saves the thresholded image.
    All operations are done within a directory named 'cropped_images', which is created if it does not exist.

    :param rgb_cropped_image: A PIL Image object representing the cropped image to process.
    :param data_cropped_path: A string representing the path to the directory where images will be saved.

    :return: The file path of the saved thresholded image.
    """
    # Create the directory if it doesn't exist
    directory_name = f"{data_cropped_path}/cropped_images"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Define file paths
    cropped_img_path = os.path.join(directory_name, "cropped_table_image.png")
    thresholded_img_path = os.path.join(directory_name, "thresholded_image.png")

    # Save the cropped image
    rgb_cropped_image.save(cropped_img_path)

    # Read the image back using OpenCV
    img1 = cv2.imread(cropped_img_path)

    # Apply threshold
    _, thresh = cv2.threshold(img1, 170, 255, cv2.THRESH_BINARY)

    # Save the thresholded image
    cv2.imwrite(thresholded_img_path, thresh)
    return thresholded_img_path


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
        # a_point_x, a_point_y = item[0][0][0], item[0][0][1]
        # c_point_x, c_point_y = item[0][2][0], item[0][2][1]
        # avg_x, avg_y = (a_point_x + c_point_x) / 2, (a_point_y + c_point_y) / 2

        # Uncomment and modify the conditional below if filtering is needed
        # if (x_range is None or x_range[0] < avg_x < x_range[1]) and (y_range is None or y_range[0] < avg_y < y_range[1]):
        bbox.append(item[0])
        text_blocks.append(item[1][0])

    y0ss = [sublist[0][1] for sublist in bbox]
    integer_list = [int(value) for value in y0ss]

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
    """
    Identifies unique horizontal lines (Y coordinates) from a dataframe that contains Y0 and Y1 coordinates for each row.
    The function calculates the average row height and uses it to determine a sensitivity threshold for detecting unique lines.

    :param df: A pandas DataFrame containing at least two columns 'Y0' and 'Y1' which represent the Y-coordinate bounds of each row.

    :return: A list of unique Y0 coordinates, representing the starting positions of unique lines.
    """
    # Calculate heights and their mean
    heights = [row["Y1"] - row["Y0"] for i, row in df.iterrows()]
    height_mean = sum(heights) / len(heights)

    # Initialize list to keep track of unique lines
    h_lines = []
    sensitivity = height_mean / 1.6

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
    """
    Identifies unique vertical lines (X coordinates) from a dataframe that contains X0 and X1 coordinates for each row.
    The function calculates the average column width and uses it to determine a sensitivity threshold for detecting unique lines.

    :param df: A pandas DataFrame containing at least two columns 'X0' and 'X1' which represent the X-coordinate bounds of each column.

    :return: A list of unique X0 coordinates, representing the starting positions of unique vertical lines.
    """

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
    """
    Assigns row and column numbers to each item in a dataframe based on its X and Y coordinates.
    The function calculates midpoints for the X and Y coordinates and uses them to determine
    which row and column each item belongs to by comparing with predefined row and column lines.

    :param df: A pandas DataFrame containing at least the columns 'X0', 'X1', 'Y0', and 'Y1'.
               These represent the bounding box coordinates of the items.
    :param row_lines: A list of Y0 values that define the horizontal lines for each row.
    :param column_lines: A list of X0 values that define the vertical lines for each column.

    :return: The updated DataFrame with 'RowNumber' and 'ColumnNumber' columns assigned.
    """
    # Calculate midpoints
    df["MidX"] = (df["X0"] + df["X1"]) / 2
    df["MidY"] = (df["Y0"] + df["Y1"]) / 2

    # Determine row and column numbers
    df["RowNumber"] = df["MidY"].apply(lambda y: find_segment(y, row_lines))
    df["ColumnNumber"] = df["MidX"].apply(lambda x: find_segment(x, column_lines))

    return df


def save_pdf_pages_as_images(pdf_path, page_numbers, output_folder):
    """
    Saves specific pages of a PDF as images. Each page is rendered at 300 DPI and saved as a PNG file.

    :param pdf_path: The file path to the input PDF.
    :param page_numbers: A list of page numbers to be saved as images. Pages are 0-indexed.
    :param output_folder: The folder where the images will be saved.
    """
    doc = fitz.open(pdf_path)

    # Loop through the specified page numbers
    for page_number in page_numbers:
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


def create_output_folder(output_folder_name):
    """
    Creates an output folder in the current working directory if it doesn't already exist.

    :param output_folder_name: The name of the folder to be created.
    :return: The full path of the output folder.
    """
    # Get the root directory
    root_directory = os.getcwd()  # This gets the current working directory

    # Concatenate the root directory with the output folder name
    output_folder = os.path.join(root_directory, output_folder_name)

    # Check if the output folder exists
    if not os.path.exists(output_folder):
        # If it doesn't exist, create the folder
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created successfully.")
    else:
        print(f"Output folder '{output_folder}' already exists.")
    return output_folder


def process_table_cells(cells):
    """
    Processes a list of table cells, filters and adjusts their bounding boxes, and resolves overlaps.

    :param cells: A list of table cell dictionaries, each containing a 'label' and a 'bbox' (bounding box),
                  along with other metadata such as 'score' for overlap resolution.

    :return: A tuple containing filtered and processed cells:
        - table_columns_only: Filtered list of cells labeled as 'table column'.
        - table_columns_header_only: Adjusted list of cells labeled as 'table column header'.
        - table_columns_row_header_only: Filtered list of cells labeled as 'table projected row header'.
        - table_columns_spanning_only_sorted: Sorted list of cells labeled as 'table spanning cell' by their y-axis position.
    """
    # Filter cells by their labels
    table_columns_only = [cell for cell in cells if cell["label"] == "table column"]
    table_columns_header_only = [
        cell for cell in cells if cell["label"] == "table column header"
    ]
    table_columns_row_header_only = [
        cell for cell in cells if cell["label"] == "table projected row header"
    ]
    table_columns_spanning_only = [
        cell for cell in cells if cell["label"] == "table spanning cell"
    ]

    # Filter spanning cells based on y-axis boundaries
    _, y_init, _, y_down = table_columns_header_only[0]["bbox"]
    table_columns_spanning_only = [
        cell for cell in table_columns_spanning_only if cell["bbox"][3] <= y_down
    ]

    # Rule 1: Adjust column header bbox based on row header overlap
    column_header_bbox = list(
        table_columns_header_only[0]["bbox"]
    )  # Ensure it's mutable
    for row_header in table_columns_row_header_only:
        row_header_bbox = row_header["bbox"]
        if (
            row_header_bbox[1] <= column_header_bbox[3]
            and row_header_bbox[3] >= column_header_bbox[1]
        ):
            overlap_top = max(column_header_bbox[1], row_header_bbox[1])
            column_header_bbox[3] = overlap_top

    table_columns_header_only[0]["bbox"] = column_header_bbox

    # Rule 2: Handle overlapping columns
    def overlap_percentage(range1, range2):
        """Calculate the percentage of overlap between two ranges."""
        max_start = max(range1[0], range2[0])
        min_end = min(range1[1], range2[1])
        overlap = max(0, min_end - max_start)
        range1_length = range1[1] - range1[0]
        return (overlap / range1_length) * 100 if range1_length != 0 else 0

    overlap_threshold = 60  # Percentage
    indices_with_overlap = []
    for i in range(len(table_columns_only)):
        for j in range(i + 1, len(table_columns_only)):
            overlap = overlap_percentage(
                [table_columns_only[i]["bbox"][0], table_columns_only[i]["bbox"][2]],
                [table_columns_only[j]["bbox"][0], table_columns_only[j]["bbox"][2]],
            )
            if overlap > overlap_threshold:
                indices_with_overlap.append((i, j))

    # Resolve overlapping columns based on scores
    for i, j in indices_with_overlap:
        if table_columns_only[i]["score"] > table_columns_only[j]["score"]:
            table_columns_only[j] = 0  # Mark for removal
        else:
            table_columns_only[i] = 0  # Mark for removal
    table_columns_only = [cell for cell in table_columns_only if cell != 0]

    # Sort spanning cells by their y-axis start position
    table_columns_spanning_only_sorted = sorted(
        table_columns_spanning_only, key=lambda x: x["bbox"][1]
    )

    return (
        table_columns_only,
        table_columns_header_only,
        table_columns_row_header_only,
        table_columns_spanning_only_sorted,
    )


def oriented_adjust_bounding_boxes(
    table_columns,
    table_headers,
    row_headers,
    spanning_headers,
    offset_bbox,
    crop_padding,
):
    """
    Adjusts bounding boxes for table columns, headers, row headers, and spanning headers.

    Parameters:
    - table_columns: List of table column cells with their bounding boxes.
    - table_headers: List of table column header cells with their bounding boxes.
    - row_headers: List of table row header cells with their bounding boxes.
    - spanning_headers: List of table spanning header cells with their bounding boxes.
    - objects: List of detected objects in the image, used to adjust bounding box positions.
    - crop_padding: Padding value used to adjust bounding boxes.

    Returns:
    - A dictionary containing lists of adjusted bounding boxes for columns, column headers,
      row headers, and spanning headers.
    """

    def adjust_lines(cells, offset_bbox, padding):
        xo, yo, x1, y1 = [], [], [], []
        for cell in cells:
            xo.append(cell["bbox"][0])
            yo.append(cell["bbox"][1])
            x1.append(cell["bbox"][2])
            y1.append(cell["bbox"][3])
        return xo, yo, x1, y1

    def transpose(matrix, condition_row=None):
        transposed_matrix = list(map(list, zip(*matrix)))
        if condition_row is not None:
            sorted_transposed_matrix = sorted(transposed_matrix, key=lambda x: x[1])
        else:
            sorted_transposed_matrix = sorted(transposed_matrix, key=lambda x: x[0])
        return sorted_transposed_matrix

    # offset_bbox = objects[0]['bbox']

    xo_column_lines, yo_column_lines, x1_column_lines, y1_column_lines = adjust_lines(
        table_columns, offset_bbox, crop_padding
    )
    (
        xo_column_header_lines,
        yo_column_header_lines,
        x1_column_header_lines,
        y1_column_header_lines,
    ) = adjust_lines(table_headers, offset_bbox, crop_padding)
    (
        xo_row_header_lines,
        yo_row_header_lines,
        x1_row_header_lines,
        y1_row_header_lines,
    ) = adjust_lines(row_headers, offset_bbox, crop_padding)
    (
        xo_spanning_header_lines,
        yo_spanning_header_lines,
        x1_spanning_header_lines,
        y1_spanning_header_lines,
    ) = adjust_lines(spanning_headers, offset_bbox, crop_padding)

    sorted_column_matrix = transpose(
        [xo_column_lines, yo_column_lines, x1_column_lines, y1_column_lines]
    )
    sorted_column_header_matrix = transpose(
        [
            xo_column_header_lines,
            yo_column_header_lines,
            x1_column_header_lines,
            y1_column_header_lines,
        ]
    )
    sorted_row_header_matrix = transpose(
        [
            xo_row_header_lines,
            yo_row_header_lines,
            x1_row_header_lines,
            y1_row_header_lines,
        ],
        condition_row=1,
    )
    sorted_spanning_header_matrix = transpose(
        [
            xo_spanning_header_lines,
            yo_spanning_header_lines,
            x1_spanning_header_lines,
            y1_spanning_header_lines,
        ]
    )
    return (
        sorted_column_matrix,
        sorted_column_header_matrix,
        sorted_row_header_matrix,
        sorted_spanning_header_matrix,
    )


def extract_blocks_with_coords(page):
    """
    Extracts text blocks along with their coordinates (bounding box) from a page.

    Parameters:
    page (object): The page object that contains the text and visual elements.

    Returns:
    tuple: A tuple containing:
        - text_blocks (list of str): A list of text blocks (words) from the page.
        - coords (list of tuples): A list of coordinates (x0, y0, x1, y1) for each word.
    """
    text_blocks, coords = [], []
    for word in page.get_text("words"):
        coords.append((word[0], word[1], word[2], word[3]))
        text_blocks.append(word[4])
    return text_blocks, coords


def find_near_duplicates(sequence, diff=5):
    """
    Finds near-duplicate values in a sequence based on a difference threshold.

    Parameters:
    sequence (list of float): The input sequence of values.
    diff (int): The threshold difference used to group near-duplicates (default is 5).

    Returns:
    defaultdict: A dictionary where keys are rounded groups of values and values
                 are lists of indices from the original sequence.
    """
    updated_tally = defaultdict(list)
    for index, item in enumerate(sequence):
        updated_tally[round(item / diff) * diff].append(index)
    return updated_tally


def remove_y_duplicates(lines, threshold=2):
    """
    Removes duplicate lines based on their Y-axis coordinates, keeping only one line
    if two lines are close to each other vertically.

    Parameters:
    lines (list of tuples): A list of lines, where each line is represented as a tuple
                            of coordinates ((x0, y0), (x1, y1)).
    threshold (float): The minimum Y-axis distance between two lines to consider them as separate.
                       Lines closer than this value are considered duplicates (default is 2).

    Returns:
    list: A cleaned list of lines, where duplicates (based on Y-axis proximity) are removed.
    """
    cleaned_lines, previous_line = [], None
    for line in sorted(lines, key=lambda line: (line[0][1] + line[1][1]) / 2):
        current_y_avg = (line[0][1] + line[1][1]) / 2
        if (
            previous_line is None
            or abs(current_y_avg - (previous_line[0][1] + previous_line[1][1]) / 2)
            > threshold
        ):
            cleaned_lines.append(line)
        previous_line = line
    return cleaned_lines


def average_whitespace_distance_by_line(page):
    """
    Calculates the average whitespace distance between words for each line of text on a PDF page.

    :param page: A PDF page object with text data, where each word is represented as a tuple containing its bounding box and text.
    :return: A dictionary where keys are line bounding boxes (y0, y1) and values are the average whitespace distance between words in that line.
    """
    lines, avg_distances = {}, {}
    for word in page.get_text("words"):
        _, y0, _, y1 = word[:4]
        for line in lines.keys():
            if not (y1 < line[0] or y0 > line[1]):
                lines[line].append(word)
                break
        else:
            lines[(y0, y1)] = [word]
    for line, words in lines.items():
        distances = [
            words[i + 1][0] - words[i][2]
            for i in range(len(words) - 1)
            if words[i + 1][0] - words[i][2] > 0
        ]
        avg_distances[line] = sum(distances) / len(distances) if distances else 0
    return avg_distances


def detect_table(page, text_blocks, count_lines, perc_vertical, perc_horizontal):
    """
    Detects if a page contains a table based on whitespace distances, numeric content,
    and line orientation ratios.

    Parameters:
    page (object): The page object that contains the text and visual elements of the page.
    text_blocks (list of str): A list of text blocks from the page.
    count_lines (int): The number of lines present on the page.
    perc_vertical (float): The percentage of vertical lines on the page.
    perc_horizontal (float): The percentage of horizontal lines on the page.

    Returns:
    bool: Returns True if the page likely contains a table, False otherwise.
    """
    average_distance = list(average_whitespace_distance_by_line(page).values())
    page_text = page.get_text()
    page_searchable = page_text.strip()
    if not page_searchable:
        return True
    if not average_distance:
        return False
    numeric_content_count = sum(
        1 for block in text_blocks if any(char.isdigit() for char in block)
    )
    return (
        len([d for d in average_distance if d > 7]) / len(average_distance) > 0.10
        or perc_vertical >= 0.8 * perc_horizontal
        and count_lines >= 3
        or numeric_content_count / len(text_blocks) > 0.15
    )


def extract_row_headers(updated_df, mapped_df, distance_threshold=65, column_number=1):
    """
    Extracts row headers from a dataframe by identifying rows that are missing large horizontal gaps
    between text blocks and merging the corresponding text entries from another dataframe.

    Parameters:
    updated_df (pd.DataFrame): The dataframe containing row and column data, including coordinates (X0, X1).
    mapped_df (pd.DataFrame): The dataframe containing text data to map and merge into the final headers.
    distance_threshold (int): Threshold for considering gaps between text blocks as large. Default is 65.
    column_number (int): The column number to filter rows for. Default is column 1.

    Returns:
    dict: A dictionary with row numbers as keys and their refined header text as values.
    """
    # Step 1: Sort and calculate distances
    df = updated_df.copy()
    df = df.sort_values(by=["RowNumber", "X0"])
    df["NextX0"] = df.groupby("RowNumber")["X0"].shift(-1)
    df["Distance"] = df["NextX0"] - df["X1"]
    df.dropna(subset=["Distance"], inplace=True)
    rows_with_large_distance = df[df["Distance"] > 1.2 * (df["Distance"].median())][
        "RowNumber"
    ].unique()

    # Step 2: Identify missing row numbers
    min_row = df["RowNumber"].min()
    max_row = df["RowNumber"].max()
    full_row_range = set(range(min_row, max_row + 1))
    missing_row_numbers = full_row_range - set(rows_with_large_distance)

    # Step 3: Filter rows in specific column
    df = updated_df.copy()
    column_1_df = df[df["ColumnNumber"] == column_number]
    rows_in_column_1 = column_1_df["RowNumber"].unique()

    # Step 4: Find intersection
    intersection = sorted(set(missing_row_numbers).intersection(set(rows_in_column_1)))

    # Step 5: Merge text entries
    nm = {}
    for index in intersection:
        merged_text = " ".join(map(str, mapped_df.iloc[index - 1].values))
        nm[index] = merged_text.strip()

    # Step 6: Refine headers
    def refine_headers(header_dict):
        refined_headers = {}
        for key, value in header_dict.items():
            # Strip leading non-alphabetic characters to find the first alphabetic character
            stripped_value = value.lstrip(
                "()[]{}<>!@#$%^&*_-+=|\\:;'\",.?/~`0123456789 "
            )

            # Check if the first alphabetic character is uppercase or if there's no alphabetic character, let it pass
            if (
                stripped_value
                and stripped_value[0].isupper()
                or not any(char.isalpha() for char in value)
            ):
                # Ensure the whole string is not just a number unless it's intended to be
                if not value.replace(" ", "").replace("(", "").replace(
                    ")", ""
                ).isdigit() or not any(char.isalpha() for char in value):
                    refined_headers[key] = value
        return refined_headers

    refined_headers = refine_headers(nm)
    return refined_headers


def get_raw_dataframes(df, pdf_column_matrix):
    """
    Processes a DataFrame by splitting text into individual words, adjusting X0 and X1 coordinates
    for each word, and mapping the text into a row-column structure based on the provided
    pdf_column_matrix.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'Text', 'X0', 'X1', and positional information.
    pdf_column_matrix (list of lists): A matrix containing column boundaries extracted from a PDF.

    Returns:
    tuple: A tuple containing the following DataFrames and data:
        - new_df (pd.DataFrame): DataFrame where text has been split and coordinates adjusted.
        - updated_df (pd.DataFrame): DataFrame with row and column numbers assigned.
        - mapped_df (pd.DataFrame): Pivoted DataFrame where text is grouped by row and column.
        - row_lines (list): A sorted list of unique row boundaries.
    """
    # Function to split text and adjust X0, X1 values based on word boundaries
    def split_text_adjust_coordinates(row):
        text = row["Text"]
        parts = text.split()  # Split text based on whitespace

        if len(parts) > 1:
            total_length = len(text)
            new_rows = []
            current_x0 = row["X0"]

            for part in parts:
                part_length = len(part)
                proportion = part_length / total_length
                x1_new = current_x0 + proportion * (row["X1"] - row["X0"])

                new_row = row.copy()
                new_row["X0"] = current_x0
                new_row["X1"] = x1_new
                new_row["Text"] = part

                new_rows.append(new_row)

                # Update current_x0 to the new starting point, adding the length of the space
                current_x0 = x1_new + ((row["X1"] - row["X0"]) / total_length)

            return new_rows
        else:
            return [row]

    # Convert text data to string
    df["Text"] = df["Text"].apply(str)

    # Apply the function to each row and create new rows
    new_rows = []
    for idx, row in df.iterrows():
        new_rows.extend(split_text_adjust_coordinates(row))

    # Create a new DataFrame from the exploded rows
    new_df = pd.DataFrame(new_rows)
    new_df.reset_index(drop=True, inplace=True)

    # Finding unique lines for rows and sorting them
    row_lines = find_unique_lines_row(new_df)
    row_lines.sort()

    # Extracting matrix for columns and sorting
    matrix = []
    for i in range(len(pdf_column_matrix)):
        matrix.append(pdf_column_matrix[i][2])
    column_lines = matrix
    column_lines.insert(0, 4)
    column_lines.sort()

    # Assign row and column numbers based on defined lines
    updated_df = assign_row_col_numbers(new_df, row_lines, column_lines)

    # Group by RowNumber and ColumnNumber and concatenate texts
    grouped = (
        updated_df.groupby(["RowNumber", "ColumnNumber"])["Text"]
        .agg(" ".join)
        .reset_index()
    )

    # Pivot the DataFrame to create a mapped view
    mapped_df = grouped.pivot(index="RowNumber", columns="ColumnNumber", values="Text")

    # Fill NaN with empty strings if needed
    mapped_df = mapped_df.fillna("")

    return new_df, updated_df, mapped_df, row_lines


def is_float(value):
    """
    Determines if a given value contains a numeric fragment that can be considered a float,
    while ensuring that the string is not predominantly alphabetic.

    Parameters:
    value (str/int/float): The value to check if it represents a float.

    Returns:
    bool: True if the value contains a valid float or numeric fragment, False otherwise.
    """
    # Convert value to string to ensure compatibility with regex
    value_str = str(value)

    # Extract all numeric fragments, allowing standalone numbers or numbers within brackets
    numeric_fragments = re.findall(
        r"-?\d*\.\d+|\d+|\{\s*-?\d*\.\d+\s*\}|\[\s*-?\d*\.\d+\s*\]|\(\s*-?\d*\.\d+\s*\)",
        value_str,
    )

    # Join all numeric fragments to calculate total length of numeric content
    "".join(numeric_fragments)

    # Calculate the percentage of the string that is alphabetic text
    alphabetic_characters = re.sub(
        r"[^a-zA-Z]", "", value_str
    )  # Remove non-alphabetic characters
    alphabetic_percentage = (
        len(alphabetic_characters) / len(value_str) if len(value_str) > 0 else 0
    )

    # Determine if the entry is numeric based on the proportion of numeric content and alphabetic text
    if len(numeric_fragments) > 0 and alphabetic_percentage <= 0.50:
        return True
    return False


def identify_table_type(df):
    """
    Identifies the type of a table by determining whether the majority of its columns contain numeric values.
    If more than 70% of the entries in a column are numeric, the table is classified as "Numeric".
    Otherwise, it is classified as "Unnumeric".

    Parameters:
    df (pd.DataFrame): The input dataframe containing the table data.

    Returns:
    str: "Numeric" if any column has more than 70% numeric values, otherwise "Unnumeric".
    """
    # Remove rows where all columns are empty
    df = df.loc[~(df == "").all(axis=1)]

    # Iterate through each column in the dataframe
    for column in df.columns:
        # Use a lambda function to apply the float check
        float_count = df[column].apply(lambda x: is_float(x)).mean()
        # print(f"Float percentage for column {column}: {float_count * 100:.2f}%")
        # Check if more than 70% of the column's entries are considered float
        if float_count > 0.70:
            return "Numeric"

    return "Unnumeric"


def create_mapped_views(df):
    """
    Function to group by RowNumber and ColumnNumber and create mapped views for X0 and Y0.

    Args:
    df (pd.DataFrame): DataFrame containing the columns RowNumber, ColumnNumber, X0, and Y0.

    Returns:
    tuple: A tuple containing two DataFrames, mapped_df_x and mapped_df_y for X0 and Y0 respectively.
    """
    # Group by RowNumber and ColumnNumber and take the first value for X0
    grouped_x = df.groupby(["RowNumber", "ColumnNumber"])["X0"].first().reset_index()
    # Pivot the DataFrame to create a mapped view for X0
    mapped_df_x = grouped_x.pivot(
        index="RowNumber", columns="ColumnNumber", values="X0"
    )
    # Fill NaN with empty strings if needed
    mapped_df_x = mapped_df_x.fillna("")

    # Repeat the process for Y0
    grouped_y = df.groupby(["RowNumber", "ColumnNumber"])["Y0"].first().reset_index()
    # Pivot the DataFrame to create a mapped view for Y0
    mapped_df_y = grouped_y.pivot(
        index="RowNumber", columns="ColumnNumber", values="Y0"
    )
    # Fill NaN with empty strings if needed
    mapped_df_y = mapped_df_y.fillna("")

    return mapped_df_x, mapped_df_y


def largest_less_than(numbers, target):
    """
    Finds the largest number in a list that is less than the target.

    Parameters:
    numbers (list of int/float): The list of numbers to search.
    target (int/float): The target number to compare against.

    Returns:
    int/float: The largest number in the list that is less than the target.
               Returns the target itself if no such number exists.
    """
    # Filter out numbers that are less than the target
    less_than_target = [num for num in numbers if num < target]

    # Return the maximum of these numbers, if the list is not empty
    if less_than_target:
        return max(less_than_target)
    else:
        return target  # or an appropriate value indicating no such number exists


def find_closest(numbers, target):
    """
    Finds the number in the list that is closest to the target value.

    Parameters:
    numbers (list of int/float): The list of numbers to search.
    target (int/float): The target number to find the closest match for.

    Returns:
    int/float: The number in the list closest to the target.
               If the list is empty or invalid, returns the target itself.
    """
    # Sort the list first (optional if the list is already sorted)

    if numbers is None or not isinstance(numbers, list) or len(numbers) == 0:
        return target
    numbers.sort()
    closest = numbers[0]
    if target < closest:
        return target
    for number in numbers:
        if abs(number - target) < abs(closest - target):
            closest = number
    return closest


def detect_horizontal_lines(image_path):
    """
    Detects horizontal lines in an image using edge detection and Hough Line Transform.
    Filters out lines that are close to each other to return unique horizontal lines.

    Parameters:
    image_path (str): The path to the input image.

    Returns:
    list: A sorted list of the y-coordinates of unique horizontal lines in the image.
          Returns None if the image cannot be loaded or no lines are detected.
    """
    # Load the image
    img = cv2.imread(image_path)
    # Check if image has been loaded properly
    if img is None:
        return None  # Early exit if image loading fails

    # Convert image to grayscale and detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)

    # Detect lines using HoughLinesP
    height, width = img.shape[:2]
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, 50, minLineLength=width / 1.5, maxLineGap=10
    )

    if lines is None:
        return None  # Early exit if no lines are detected

    lines = [line[0] for line in lines]

    # Filter lines to include only horizontal lines
    horizontal_lines = [line for line in lines if is_horizontal(line)]

    # Remove lines that are too close to each other
    threshold = 10  # Distance threshold to consider lines as duplicates
    unique_lines = []

    for line in horizontal_lines:
        if not unique_lines or all(
            line_distance(line, unique_line) >= threshold
            for unique_line in unique_lines
        ):
            unique_lines.append(line)

    # Extract the y-coordinates of the unique horizontal lines
    main_line = [line[1] for line in unique_lines]
    main_line.sort()

    return main_line


def line_distance(line1, line2):
    """
    Calculates the minimum distance between two lines based on their endpoints.

    Parameters:
    line1 (tuple): A tuple representing the first line's coordinates (x1, y1, x2, y2).
    line2 (tuple): A tuple representing the second line's coordinates (x3, y3, x4, y4).

    Returns:
    float: The minimum distance between the two lines by calculating the distance between
           their respective endpoints.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    return min(math.dist((x1, y1), (x3, y3)), math.dist((x2, y2), (x4, y4)))


def is_horizontal(line, angle_threshold=10):
    """
    Determines if a given line is approximately horizontal by calculating its angle
    and checking whether it falls within a specified threshold of horizontal (0 or 180 degrees).

    Parameters:
    line (tuple): A tuple representing the line's coordinates (x1, y1, x2, y2).
    angle_threshold (float): The maximum allowable deviation from 0 or 180 degrees
                             for the line to be considered horizontal. Default is 10 degrees.

    Returns:
    bool: True if the line is considered horizontal, False otherwise.
    """
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold


def combine_rows_to_header(df, count):
    """
    Combines the first 'count' rows of a DataFrame to form a single header.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    count (int): The number of rows to combine to form the header.

    Returns:
    pd.DataFrame: The DataFrame with the new header and rows below the header.
    """
    # Combine the first 'count' rows to form a single header
    new_header = df.iloc[0:count].apply(
        lambda x: " ".join(x.dropna().astype(str)), axis=0
    )

    # Set the new header
    df.columns = new_header

    # Remove the header rows from the DataFrame
    df = df[count:]

    # Reset index if necessary
    df.reset_index(drop=True, inplace=True)

    return df


def get_image_height(image_path):
    """
    Returns the height of the image at the given path.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    int: The height of the image in pixels.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height = image.shape[0]

    return height


def check_within_percentage_range(height, numbers):
    """
    Checks if any number in the list falls within 25% and 75% of the given height.

    Parameters:
    height (int): The height to calculate the range.
    numbers (list): A list of numbers to check.

    Returns:
    str: "yes" if any number falls within the range, otherwise "no".
    """
    if numbers is None or not isinstance(numbers, list) or len(numbers) == 0:
        return "no"
    lower_bound = 0.25 * height
    upper_bound = 0.75 * height

    for number in numbers:
        if lower_bound <= number <= upper_bound:
            return "yes"
    return "no"


def group_by_main_line(main_list, row_list):
    """
    Groups row_list entries by the nearest preceding main_list entry.

    Parameters:
    main_list (list): A list of main line values.
    row_list (list): A list of row line values to be grouped.

    Returns:
    dict: A dictionary where keys are the indices of the main_list entries and
          values are lists of indices of the row_list entries grouped by the
          nearest preceding main_list entry.
    """

    def find_group_index(main_list, row_val):
        """
        Finds the index of the closest preceding 'main_list' entry.

        Parameters:
        main_list (list): A list of main line values.
        row_val (int): The row value to find the preceding main line for.

        Returns:
        int: The index of the last valid 'main_list' entry that is less than or
             equal to the row_val.
        """
        # Find the last main line value that is less than or equal to the row value
        filtered = [i for i in main_list if i <= row_val]
        if not filtered:
            return None  # if no valid main line is found, which should not happen in this context
        return main_list.index(
            filtered[-1]
        )  # Return the index of the last valid 'main_list' entry

    # Group row_list by the nearest preceding main_list
    group_dict = {}
    for row in row_list:
        group_index = find_group_index(main_list, row)
        if group_index is not None:
            if group_index in group_dict:
                group_dict[group_index].append(row_list.index(row))
            else:
                group_dict[group_index] = [row_list.index(row)]

    return group_dict


def get_aggregations(df):
    """
    Generates a dictionary of aggregation functions for each column of a DataFrame.
    The aggregation logic is based on the data type of each column:

    - Numeric columns: The values are converted to strings and concatenated.
    - String columns: The values are concatenated directly.
    - Other data types: The values are also concatenated directly.

    Parameters:
    df (pd.DataFrame): The input DataFrame for which to generate aggregation functions.

    Returns:
    dict: A dictionary where the keys are column names and the values are functions that define
          how to aggregate the respective column (by concatenating the values as strings).
    """
    aggregations = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            aggregations[col] = lambda x: " ".join(str(x))
        elif pd.api.types.is_string_dtype(df[col]):
            aggregations[col] = lambda x: " ".join(x)
        else:
            aggregations[col] = lambda x: " ".join(x)
    return aggregations


def should_merge_based_on_string(text, next_text):
    """Check if the text ends with a specific string that requires merging with the next."""
    # Check for word-level end strings
    text_words = text.split()
    if text_words and any(text_words[-1] == s for s in word_end_strings):
        return True

    # Check for character-level end strings
    if any(text.endswith(s) for s in char_end_strings):
        return True

    return False


def get_merged_text_values(text_values, x_coords, y_coords):
    """
    Merges consecutive text values in a list based on certain conditions, while maintaining
    the position of empty spaces and invalid entries. Merging is controlled by x-coordinates
    and certain string conditions.

    Parameters:
    text_values (list of str): List of text values to be merged.
    x_coords (list of float): List of x-coordinates corresponding to the text values.
    y_coords (list of float): List of y-coordinates corresponding to the text values.

    Returns:
    list of str: A list of text values where certain entries have been merged.
    """
    output = [""] * len(text_values)  # Initialize output with empty strings
    i = 0

    while i < len(text_values):
        if (
            not text_values[i].strip() or x_coords[i] == "" or y_coords[i] == ""
        ):  # Skip processing if not valid
            output[i] = text_values[i]  # Preserve spaces or empty entries
            i += 1
            continue

        current_text = text_values[i]
        current_x = x_coords[i]
        y_coords[i]

        # Start merging process
        merge_text = current_text
        j = i + 1
        merge_allowed = True  # Flag to control merging based on space presence

        while j < len(text_values):
            if not text_values[j].strip() or x_coords[j] == "" or y_coords[j] == "":
                if (
                    text_values[j] == " " or text_values[j] == ""
                ):  # Check if there is a space
                    merge_allowed = False  # Prevent merging if there's a space
                j += 1
                continue

            if not merge_allowed:
                break  # Stop merging if a space was encountered before this point

            next_text = text_values[j]
            next_x = x_coords[j]
            y_coords[j]

            # Check for conditions to merge
            if (
                should_merge_based_on_string(merge_text, next_text)
                or (next_x - current_x) > 25
                or next_text[0].islower()
            ):  # Include lowercase rule
                merge_text += f" {next_text}"
            else:
                break

            j += 1

        # Assign merged text to the first index of merge group
        output[i] = merge_text
        for k in range(i + 1, j):
            output[
                k
            ] = " "  # Fill following positions with spaces if they were considered for merging

        i = j  # Move to the next group

    return output


def apply_and_join(df, df_x, df_y):
    """
    Applies the merging logic to each column of a DataFrame by utilizing corresponding x and y coordinates.
    The function defragments text entries based on certain merging conditions and returns a new DataFrame
    with the merged text.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing text values to be merged.
    df_x (pd.DataFrame): A DataFrame containing the x-coordinates corresponding to the text values.
    df_y (pd.DataFrame): A DataFrame containing the y-coordinates corresponding to the text values.

    Returns:
    pd.DataFrame: A new DataFrame where the text values have been defragmented (merged) based on the
                  conditions applied through `get_merged_text_values`.
    """
    result_dict = {}
    df_x.columns = df.columns
    df_y.columns = df.columns
    for column in df.columns:
        text_values = df[column].values
        x_coords = df_x[column].values
        y_coords = df_y[column].values

        column_defragmented = get_merged_text_values(text_values, x_coords, y_coords)
        result_dict[column] = column_defragmented

    return pd.DataFrame(result_dict)


def replace_row_header(df, row_headers):
    """
    Replaces the text in specified rows of the DataFrame with '' based on the given dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame to update.
    row_headers (dict): A dictionary where keys are row indices and values are the new text to replace.

    Returns:
    pd.DataFrame: The updated DataFrame.
    """
    for row_index in row_headers.keys():
        df.iloc[row_index - 1] = ""

    return df


# Example DataFrame
data = {
    "Column1": ["Text1", np.nan, "Text3", ""],
    "Column2": ["A", "B", "", "D"],
    "Column3": ["E", "F", "G", "H"],
}
mapped_df = pd.DataFrame(data)


def count_non_blank_cells(df):
    """
    Counts non-blank cells in each row of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    list: A list containing the count of non-blank cells for each row.
    """

    # Function to check if a cell is blank
    def is_blank(cell):
        return pd.isna(cell) or cell == "" or cell == " "

    blank_list = []
    for index, row in df.iterrows():
        blank_count = 0
        for cell in row:
            if not is_blank(cell):
                blank_count += 1
        # print(f"Row {index} has {blank_count} non-blank cells.")
        blank_list.append(blank_count)

    return blank_list


def find_intervals_max_positions(data):
    """
    Finds intervals between each pair of indices where the maximum value occurs in the list.

    Parameters:
    data (list): The list to analyze.

    Returns:
    list: A list of intervals between each pair of indices where the maximum value occurs.
    """
    # Find the maximum value in the list
    max_value = max(data)

    # Find all indices where the maximum value occurs
    max_positions = [index for index, value in enumerate(data) if value == max_value]

    # Create intervals between each pair of indices and ensure all indices are covered
    intervals = []
    if max_positions:
        # Add interval from start to first maximum position
        if max_positions[0] > 0:
            intervals.append(list(range(0, max_positions[0])))

        # Add intervals between each pair of maximum positions
        for i in range(len(max_positions) - 1):
            intervals.append(list(range(max_positions[i], max_positions[i + 1])))

        # Add interval from last maximum position to end
        intervals.append(list(range(max_positions[-1], len(data))))

    return intervals


def new_find_intervals_max_positions(data, threshold=1):
    """
    Finds intervals between each pair of indices where the significant values occur in the list.
    Significant values are those that are greater than or equal to the specified threshold.

    Parameters:
    data (list): The list to analyze.
    threshold (int, optional): The threshold to determine significant values. Defaults to 1.

    Returns:
    list: A list of intervals between each pair of indices where significant values occur.
    """
    # Find all indices where the values are greater than or equal to the threshold
    significant_positions = [
        index for index, value in enumerate(data) if value >= threshold
    ]

    # Create intervals between each pair of significant positions and ensure all indices are covered
    intervals = []
    if significant_positions:
        # Add interval from start to first significant position
        if significant_positions[0] > 0:
            intervals.append(list(range(0, significant_positions[0])))

        # Add intervals between each pair of significant positions
        for i in range(len(significant_positions) - 1):
            intervals.append(
                list(range(significant_positions[i], significant_positions[i + 1]))
            )

        # Add interval from last significant position to end
        intervals.append(list(range(significant_positions[-1], len(data))))

    return intervals


def merge_rows_generalized(df, intervals):
    """
    Merges rows in a DataFrame based on specified intervals, combining numeric and string columns
    appropriately. Numeric values can be aggregated (e.g., summed or averaged), and string values
    are concatenated while preserving order and removing duplicates.

    Parameters:
    df (pd.DataFrame): The input DataFrame with rows to be merged.
    intervals (list of lists): A list of intervals (ranges of indices) specifying which rows should be merged.

    Returns:
    pd.DataFrame: A new DataFrame where the rows within the specified intervals have been merged.
    """
    # Prepare a list to collect merged rows
    merged_rows = []

    for interval in intervals:
        if interval:
            subset = df.iloc[interval[0] : interval[-1] + 1]
            merged_row = {}

            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Use mean, sum, or first for numeric data as needed
                    merged_row[column] = " ".join(
                        dict.fromkeys(subset[column])
                    )  # Example: mean of the numeric values
                elif pd.api.types.is_string_dtype(df[column]):
                    # Concatenate unique values in the order they appear
                    merged_row[column] = " ".join(
                        dict.fromkeys(subset[column])
                    )  # Removes duplicates, preserves order
                else:
                    # Handle other types similarly, e.g., for datetime or categorical data
                    merged_row[column] = " ".join(
                        dict.fromkeys(subset[column])
                    )  # Convert to string if necessary

            merged_rows.append(merged_row)

    return pd.DataFrame(merged_rows)


def format_space_text(text):
    """
    Formats a text by splitting it into parts based on two or more spaces, and then joins
    the parts with newline characters to mimic separate lines.

    Parameters:
    text (str): The input text to be formatted. If the input is not a string, it is returned as-is.

    Returns:
    str: The formatted text where parts are separated by newlines, or the original input if it's not a string.
    """
    # Ensure the function handles non-string data gracefully
    if isinstance(text, str):
        # Use regular expression to split on two or more spaces
        parts = re.split(r"(?<!\S)\s{0,}(?!\S)", text.strip())
        # Join these parts with '\n' to create the appearance of separate lines
        formatted_text = "\n".join(parts)
        return formatted_text
    else:
        return text


def is_row_empty(row):
    """
    Function to check if a row is empty
    """
    return all(isinstance(x, str) and x.strip() == "" for x in row)


def check_uppercase_percentage(df):
    """
    Checks each column in the DataFrame to see if more than 75% of non-empty string entries
    start with an uppercase letter. If any column meets this condition, it returns "Yes",
    otherwise it returns "No".

    Parameters:
    df (pd.DataFrame): The input DataFrame to be analyzed.

    Returns:
    str: "Yes" if any column has more than 75% of entries starting with an uppercase letter,
         otherwise "No".
    """
    for column in df.columns:
        if df[column].dtype == "object":
            # Count the number of non-empty strings that start with an uppercase letter
            count_uppercase = (
                df[column]
                .apply(
                    lambda x: isinstance(x, str)
                    and len(x.strip()) > 0
                    and x.strip()[0].isupper()
                )
                .sum()
            )
            # Calculate the percentage
            percentage_uppercase = (count_uppercase / len(df[column])) * 100
            if percentage_uppercase > 75:
                return "Yes"
    return "No"


def replace_empty_column_names(df, placeholder="Placeholder"):
    """
    Replaces empty or whitespace column names in a DataFrame with a unique placeholder.

    Parameters:
    df (pd.DataFrame): The input DataFrame with possibly empty column names.
    placeholder (str): The placeholder string to use for empty column names.

    Returns:
    pd.DataFrame: The DataFrame with unique column names.
    """
    columns = []
    placeholder_count = 1

    for col in df.columns:
        if col.strip() == "":
            new_col_name = f"{placeholder}_{placeholder_count}"
            placeholder_count += 1
        else:
            new_col_name = col.strip()

        while new_col_name in columns:
            new_col_name = f"{new_col_name}_{placeholder_count}"
            placeholder_count += 1

        columns.append(new_col_name)

    df.columns = columns
    return df


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def add_dummy_rows(df):
    """
    Add dummy rows to ensure continuous row numbering in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potentially missing row numbers.

    Returns:
    pd.DataFrame: The DataFrame with added dummy rows.
    """
    # Create a range of row numbers based on the min and max values in the index
    full_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1, step=1)

    # Reindex the DataFrame to include the full range of row numbers
    df_reindexed = df.reindex(full_index)
    df_reindexed = df_reindexed.fillna("")

    return df_reindexed


def update_row_headers_result(df, group_dict, row_headers, count_increment=1):
    """
    Update the row headers in the DataFrame based on the group_dict and row_headers provided.

    Parameters:
    df (pd.DataFrame): The main DataFrame to be updated.
    group_dict (dict): A dictionary where keys are group indices and values are lists of row indices.
    row_headers (dict): A dictionary where keys are row indices and values are row header strings.
    count_increment (int): The increment value to update the row headers indices.

    Returns:
    pd.DataFrame: The updated DataFrame with inserted row headers.
    """
    # Update the row headers based on count_increment
    updated_row_headers = {
        key - count_increment: value for key, value in row_headers.items()
    }

    # Find the appropriate index in the group_dict and update the DataFrame
    for updated_index, header in updated_row_headers.items():
        group_key = None
        for k, v in group_dict.items():
            if updated_index in v:
                group_key = k
                break
        if group_key is not None and group_key in df.index:
            # Append the new header to the existing value in the first column
            if df.columns[0] in df:
                current_value = str(df.at[group_key, df.columns[0]])
                if (
                    group_dict[group_key][0] == updated_index
                ):  # Check if it's the very first instance
                    df.at[group_key, df.columns[0]] = (
                        header + " " + "\n" + current_value
                    )
                else:
                    df.at[group_key, df.columns[0]] = (
                        current_value + " " + "\n" + header
                    )
        else:
            print(
                f"Warning: Unable to find group_key {group_key} for updated_index {updated_index} in the DataFrame."
            )

    return df


def update_row_headers_with_intervals(df, intervals, row_headers, count_increment=1):
    """
    Update the row headers in the DataFrame based on the intervals and row_headers provided.

    Parameters:
    df (pd.DataFrame): The main DataFrame to be updated.
    intervals (list of lists): A list of intervals where each interval is a list of row indices.
    row_headers (dict): A dictionary where keys are row indices and values are row header strings.
    count_increment (int): The increment value to update the row headers indices.

    Returns:
    pd.DataFrame: The updated DataFrame with inserted row headers.
    """
    # Update the row headers based on count_increment
    updated_row_headers = {
        key - count_increment: value for key, value in row_headers.items()
    }

    # Find the appropriate interval and update the DataFrame
    for updated_index, header in updated_row_headers.items():
        interval_index = None
        for idx, interval in enumerate(intervals):
            if updated_index in interval:
                interval_index = idx
                break
        if interval_index is not None and interval_index in df.index:
            # Get the current value in the first column
            current_value = str(df.at[interval_index, df.columns[0]])

            # Check if it's the very first instance
            if intervals[interval_index][0] == updated_index:
                # Add header in front with \n after the inserted value
                df.at[interval_index, df.columns[0]] = header + "\n" + current_value
            else:
                # Add header at the back with \n before the inserted value
                df.at[interval_index, df.columns[0]] = current_value + "\n" + header

    return df


def insert_row_headers_result_uppercase(df, initial_row_headers, count_plus_one):
    """
    Inserts row headers into a DataFrame at specified positions, adjusting the positions based on a
    provided count offset. The function validates the input data, updates the row headers' positions,
    and then inserts the row headers into the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame into which the row headers will be inserted.
    initial_row_headers (dict): A dictionary where the keys represent the positions of the row headers
                                and the values are the row header strings.
    count_plus_one (int): The count used to adjust the positions of the row headers.

    Returns:
    pd.DataFrame: The DataFrame with the row headers inserted at the adjusted positions.
    """
    # Validate input data
    if not data:
        # print("The data dictionary is empty.")
        return df

    if not initial_row_headers:
        # print("The initial row headers dictionary is empty.")
        return df

    # Update row headers by subtracting the given count + 1
    updated_row_headers = {
        key - count_plus_one: value for key, value in initial_row_headers.items()
    }

    # Ensure that updated row headers have valid positions
    max_position = max(updated_row_headers.keys(), default=0)
    min_position = min(updated_row_headers.keys(), default=0)

    if min_position < 0 or max_position > df.index[-1] + 1:
        # print("Row header positions are out of the valid range.")
        return df

    # Convert the dataframe to a list of lists for easier manipulation
    df_list = df.values.tolist()

    # Insert the row headers at the specified positions
    num_columns = df.shape[1]
    for pos, header in sorted(updated_row_headers.items()):
        df_list.insert(pos, [header] + [""] * (num_columns - 1))

    # Convert the list back to a dataframe
    updated_df = pd.DataFrame(df_list, columns=df.columns)

    return updated_df


def process_page_characteristics(image, model, device, page):
    """
    Processes the characteristics of a page, including object detection, text extraction,
    bounding box analysis, and page metadata such as rotation and searchability.

    Parameters:
    image (PIL.Image or similar): The image of the page to be processed.
    file_path (str): The file path for the document.
    model (torch.nn.Module or similar): The model used for object detection.
    device (torch.device): The device (CPU/GPU) used for running the model.
    page (Page object): The page object containing text and bounding box information.

    Returns:
    tuple: A tuple containing the following:
        - page_table_rotated (bool): Whether the table on the page is detected as rotated.
        - page_mix_ocr (bool): Whether any bounding boxes contain 'ignore-text', indicating mixed OCR.
        - page_rotation (int): The rotation angle of the page.
        - page_searchable (bool): Whether the page contains searchable text.
        - objects (list): A list of detected objects sorted by their vertical position.
        - structure_transform (function): The transformation used for structure detection.
    """

    # Get transformations for detection and structure
    detection_transform, structure_transform = get_transform()

    # Detect objects using the model and transform
    objects = get_objects(image, model, detection_transform, device)

    # Initialize page_table_rotated flag
    if objects:
        page_table_rotated = objects[0]["label"] == "table rotated"
    else:
        page_table_rotated = False

    # Extract text and bounding boxes from the page
    page_text = page.get_text()
    bbox_log = page.get_bboxlog()

    # Check if any bounding boxes have 'ignore-text'
    page_mix_ocr = any("ignore-text" in bbox[0] for bbox in bbox_log)

    # Get page rotation
    page_rotation = page.rotation

    # Check if the page text is searchable
    page_searchable = bool(page_text.strip())

    # Sort objects by their vertical position (second coordinate in bbox)
    objects = sorted(objects, key=lambda x: x["bbox"][1])

    # Return all extracted information as a dictionary
    return (
        page_table_rotated,
        page_mix_ocr,
        page_rotation,
        page_searchable,
        objects,
        structure_transform,
    )


def process_table_image(
    tables_crops,
    i,
    device,
    structure_transform,
    structure_model,
    outputs_to_objects,
    ocr,
    crop_padding,
    data_cropped_path,
    objects,
):
    """
    Process a cropped table image and extract structured data.

    :param tables_crops: List of cropped tables.
    :param i: Index of the current table to process.
    :param device: Computational device (CPU/GPU) for processing.
    :param structure_transform: Image transformations for preprocessing.
    :param structure_model: Model for structure prediction.
    :param outputs_to_objects: Function to convert model outputs to objects.
    :param ocr: OCR engine for text recognition.
    :param crop_padding: Padding used during the cropping process.
    :return: Tuple containing processed DataFrames and row headers.
    """
    # Process the oriented cropped table image
    cropped_table = tables_crops[i]["image"]
    process_table_image, rgb_cropped_image = oriented_process_cropped_table_image(
        cropped_table, device, structure_transform, structure_model, outputs_to_objects
    )

    # Get table cells classified by their type and adjusted bounding boxes
    (
        table_columns_only,
        table_columns_header_only,
        table_columns_row_header_only,
        table_columns_spanning_only_sorted,
    ) = process_table_cells(process_table_image)
    (
        pdf_column_matrix,
        pdf_column_header_matrix,
        pdf_row_header_matrix,
        pdf_spanning_header_matrix,
    ) = oriented_adjust_bounding_boxes(
        table_columns_only,
        table_columns_header_only,
        table_columns_row_header_only,
        table_columns_spanning_only_sorted,
        objects[i]["bbox"],
        crop_padding,
    )

    # Save the cropped image for OCR and perform OCR
    ocr_img_path = process_and_save_cropped_image(rgb_cropped_image, data_cropped_path)
    result = ocr.ocr(ocr_img_path, cls=True)

    # Process OCR results
    df, integer_list = process_ocr_results(result)
    " ".join(df["Text"].values)

    # Generate raw and mapped DataFrames
    new_df, updated_df, mapped_df, row_lines = get_raw_dataframes(df, pdf_column_matrix)
    mapped_df = add_dummy_rows(mapped_df)
    row_headers = extract_row_headers(updated_df, mapped_df)

    return (
        new_df,
        updated_df,
        mapped_df,
        row_lines,
        row_headers,
        pdf_column_header_matrix,
    )


def process_numeric_table(image_path, pdf_column_header_matrix, row_lines, mapped_df):
    """
    Process a table image to detect horizontal lines, find the closest header, and map rows to headers.

    Parameters:
    image_path (str): Path to the cropped table image.
    pdf_column_header_matrix (list): 2D list or array containing the coordinates of the headers in the PDF.
    row_lines (list): List of y-coordinates representing the row lines in the image.
    mapped_df (DataFrame): DataFrame that has been mapped but needs to be combined with rows to header.

    Returns:
    DataFrame: A new DataFrame where rows are combined with the detected header.
    """
    # Detect horizontal lines in the image
    CV_y_coordinates = detect_horizontal_lines(image_path)

    # Find the closest header y-coordinate using euclidean distance
    header_y_coordinate = find_closest(CV_y_coordinates, pdf_column_header_matrix[0][3])

    # Count how many row lines are before the header
    count = sum(1 for i in row_lines if i <= (header_y_coordinate + 1))

    # Combine the rows with the corresponding header
    new_mapped_df = combine_rows_to_header(mapped_df, count)

    # Create a copy of the new mapped DataFrame
    new_main_df = new_mapped_df.copy()

    return new_main_df


def create_mapped_df(new_df, mapped_df, row_headers):
    """
    Provide mapped dataframes and returns the updated dataframes.

    Parameters:
    new_df (pd.DataFrame): A new dataframe used to create mapped views.
    mapped_df (pd.DataFrame): The original dataframe with existing row headers.
    row_headers (list): A list of new row headers to replace in the dataframes.

    Returns:
    tuple: A tuple containing the updated mapped_df, mapped_df_x, and mapped_df_y.
    """
    mapped_df_x, mapped_df_y = create_mapped_views(new_df)
    mapped_df = replace_row_header(mapped_df, row_headers)
    mapped_df_x = replace_row_header(mapped_df_x, row_headers)
    mapped_df_y = replace_row_header(mapped_df_y, row_headers)
    return mapped_df, mapped_df_x, mapped_df_y


def analyze_image_y_coordinates(image_path, pdf_column_header_matrix, row_lines):
    """
    Analyzes the y-coordinates of detected horizontal lines in the image and determines the closest header
    y-coordinate using Euclidean distance. It also counts the number of row lines above or near the determined
    header coordinate.

    Parameters:
    image_path (str): The path to the image file for processing.
    pdf_column_header_matrix (list): A matrix containing the y-coordinates of column headers from a PDF.
    row_lines (list): A list of y-coordinates representing horizontal row lines.

    Returns:
    tuple: A tuple containing the list of detected y-coordinates (CV_y_coordinates) and the count of row lines
           above or near the determined header coordinate.
    """
    CV_y_coordinates = detect_horizontal_lines(image_path)

    header_y_coordinate = find_closest(CV_y_coordinates, pdf_column_header_matrix[0][3])

    if (pdf_column_header_matrix[0][3] - header_y_coordinate) > 30:
        new_header_y_coordinate = pdf_column_header_matrix[0][3]
    else:
        new_header_y_coordinate = header_y_coordinate
    count = sum(1 for i in row_lines if i <= (new_header_y_coordinate + 1))
    return CV_y_coordinates, count


def update_mapped_df(mapped_df, mapped_df_x, mapped_df_y, count):
    """
    Updates the provided dataframes by combining rows up to a specified count
    into headers and replacing empty column names with default values.

    Parameters:
    mapped_df (pd.DataFrame): The original dataframe with rows to be combined into headers.
    mapped_df_x (pd.DataFrame): A secondary dataframe with rows to be combined into headers.
    mapped_df_y (pd.DataFrame): Another dataframe with rows to be combined into headers.
    count (int): The number of rows to combine into the header.

    Returns:
    tuple: A tuple containing the updated dataframes (new_mapped_df, new_mapped_df_x, new_mapped_df_y).
    """

    new_mapped_df = combine_rows_to_header(mapped_df, count)
    new_mapped_df = replace_empty_column_names(new_mapped_df)
    new_mapped_df_x = combine_rows_to_header(mapped_df_x, count)
    new_mapped_df_x = replace_empty_column_names(new_mapped_df_x)
    new_mapped_df_y = combine_rows_to_header(mapped_df_y, count)
    new_mapped_df_y = replace_empty_column_names(new_mapped_df_y)
    return new_mapped_df, new_mapped_df_x, new_mapped_df_y


def process_grouped_dataframe(CV_y_coordinates, new_row_lines, new_mapped_df):
    """
    Groups rows in the dataframe based on vertical coordinates and performs aggregations.

    Parameters:
    CV_y_coordinates (list): A list of y-coordinates representing detected lines in the image.
    new_row_lines (list): A list of y-coordinates for new row lines to assist in grouping.
    new_mapped_df (pd.DataFrame): The dataframe to be processed and grouped.

    Returns:
    tuple: A tuple containing:
        - new_main_df (pd.DataFrame): The dataframe with grouped and aggregated data.
        - group_dict (dict): A dictionary mapping group IDs to the respective row indices.
    """
    group_dict = group_by_main_line(CV_y_coordinates, new_row_lines)
    df = new_mapped_df.copy()
    # Create a new column for group ID based on index
    group_ids = {idx: group for group, indices in group_dict.items() for idx in indices}
    df["GroupID"] = df.index.map(group_ids)
    # Exclude the GroupID from aggregation
    aggregations = get_aggregations(df.drop("GroupID", axis=1))
    # Group by the 'GroupID' and aggregate
    new_main_df = df.groupby("GroupID").agg(aggregations).reset_index(drop=True)
    # Apply the formatting function to every column in the dataframe
    for column in new_main_df.columns:
        new_main_df[column] = new_main_df[column].apply(format_space_text)
    return new_main_df, group_dict


def process_dataframe_intervals(result_df, row_headers, count):
    """
    Processes the dataframe by identifying intervals of non-blank cells, merging rows based on those intervals,
    and updating row headers accordingly.

    Parameters:
    result_df (pd.DataFrame): The dataframe to be processed.

    Returns:
    pd.DataFrame: The processed dataframe with merged rows and updated headers.
    """
    blank_list = count_non_blank_cells(result_df)
    intervals = new_find_intervals_max_positions(blank_list, threshold=2)
    new_main_df = merge_rows_generalized(result_df, intervals)
    for column in new_main_df.columns:
        new_main_df[column] = new_main_df[column].apply(format_space_text)
    new_main_df = update_row_headers_with_intervals(
        new_main_df, intervals, row_headers, count_increment=(count + 1)
    )
    return new_main_df


def df_to_pickle(new_main_df, temp_output_folder, page_num, maxi, rotated):
    """
    Saves the given DataFrame as a pickle file in the specified output folder. The file name is based on
    the page number, index of the table, and whether the table is rotated.

    Parameters:
    new_main_df (pd.DataFrame): The DataFrame to be saved.
    temp_output_folder (str): Path to the folder where the pickle file will be saved.
    page_num (int): The page number where the table was extracted from.
    maxi (int): The index of the table on the current page.
    rotated (str): A flag ("yes" or "no") indicating whether the table was rotated.

    Returns:
    None: The function saves the DataFrame as a pickle file.
    """
    if rotated == "yes":
        new_main_df.to_pickle(
            temp_output_folder
            + "df_oriented"
            + "_"
            + str(page_num)
            + "_"
            + str(maxi)
            + ".pkl"
        )
    else:
        new_main_df.to_pickle(
            temp_output_folder + "df" + "_" + str(page_num) + "_" + str(maxi) + ".pkl"
        )


def process_non_numeric_table(
    image_path,
    pdf_column_header_matrix,
    row_lines,
    new_df,
    mapped_df,
    row_headers,
    temp_output_folder,
    page_num,
    maxi,
    rotated,
):
    """
    Processes a non-numeric table, analyzing its structure, applying transformations to mapped data,
    and saving the result based on certain conditions (uppercase percentage, grouping, etc.).

    Parameters:
    image_path (str): Path to the image of the table.
    pdf_column_header_matrix (list): The matrix containing the column headers of the table.
    row_lines (list): List of y-coordinates representing the row lines in the table.
    new_df (pd.DataFrame): The DataFrame containing the original data for processing.
    mapped_df (pd.DataFrame): The mapped DataFrame representing the extracted table structure.
    row_headers (list): List of row headers for the table.
    temp_output_folder (str): Path to the temporary output folder for saving results.
    page_num (int): The page number where the table is located.
    maxi (int): The index of the current table being processed.
    rotated (str): Indicator whether the table is rotated ("yes" or "no").

    Returns:
    None: The function saves the processed DataFrame as a pickle file.
    """
    CV_y_coordinates, count = analyze_image_y_coordinates(
        image_path, pdf_column_header_matrix, row_lines
    )
    mapped_df, mapped_df_x, mapped_df_y = create_mapped_df(
        new_df, mapped_df, row_headers
    )
    new_mapped_df, new_mapped_df_x, new_mapped_df_y = update_mapped_df(
        mapped_df, mapped_df_x, mapped_df_y, count
    )
    new_row_lines = [x + 3 for x in row_lines[count:]]
    height = get_image_height(image_path)
    result = check_within_percentage_range(height, CV_y_coordinates)
    if len(new_row_lines) <= 5:
        result = "no"
    if result == "yes":
        new_main_df, group_dict = process_grouped_dataframe(
            CV_y_coordinates, new_row_lines, new_mapped_df
        )
        new_main_df = update_row_headers_result(
            new_main_df, group_dict, row_headers, count_increment=(count + 1)
        )
        df_to_pickle(new_main_df, temp_output_folder, page_num, maxi, rotated)
    else:
        result_df = apply_and_join(new_mapped_df, new_mapped_df_x, new_mapped_df_y)
        filtered_df = result_df[~result_df.apply(is_row_empty, axis=1)]
        result_uppercase = check_uppercase_percentage(filtered_df)
        if result_uppercase == "Yes":
            new_main_df = filtered_df.copy()
            count_plus_one = count + 1
            new_main_df = insert_row_headers_result_uppercase(
                new_main_df, row_headers, count_plus_one
            )
            df_to_pickle(new_main_df, temp_output_folder, page_num, maxi, rotated)
        else:
            new_main_df = process_dataframe_intervals(result_df, row_headers, count)
            df_to_pickle(new_main_df, temp_output_folder, page_num, maxi, rotated)


def extract_table(
    Pages_Contains_Tables,
    ocr,
    evaluate_dict,
    main_evaluate_dict,
    temp_output_folder,
    output_folder_image_path,
    doc,
    model,
    structure_model,
    device,
    detection_class_thresholds,
    crop_padding,
    data_cropped_path,
):
    """
    Processes pages containing tables from a document, identifying and processing numeric and non-numeric tables.
    Handles rotated, OCR-mixed tables, and object detection-based tables. Outputs the processed tables
    into specified folders based on their characteristics.

    Parameters:
    Pages_Contains_Tables (list): List of page numbers that are known to contain tables.
    ocr (object): OCR model used for text extraction.
    evaluate_dict (dict): Dictionary used for evaluation.
    main_evaluate_dict (dict): Main evaluation dictionary.
    temp_output_folder (str): Path to the temporary output folder.
    output_folder_image_path (str): Path where images of the document pages are stored.
    doc (object): Document object representing the PDF or other format.
    model (object): Object detection model used for identifying objects in the document.
    structure_model (object): Model used for processing the structure of the table.
    device (str): Device (CPU or GPU) to be used for model inference.
    detection_class_thresholds (dict): Thresholds used for detecting specific classes (e.g., tables).
    crop_padding (int): Padding to apply when cropping detected table regions.
    data_cropped_path (str): Path where cropped table images are stored.

    Returns:
    None: The function saves the processed data in files, logging the progress and errors.
    """
    for m in Pages_Contains_Tables:
        page_num = m
        file_path = output_folder_image_path + f"/page_{page_num}.png"

        page = doc.load_page(page_num)
        logging.info(f"Processing page {page_num}.")

        (
            tokens,
            xo_column_lines,
            yo_column_lines,
            x1_column_lines,
            y1_column_lines,
            xo_column_header_lines,
            yo_column_header_lines,
            x1_column_header_lines,
            y1_column_header_lines,
            xo_row_header_lines,
            yo_row_header_lines,
            x1_row_header_lines,
            y1_row_header_lines,
            xo_spanning_header_lines,
            yo_spanning_header_lines,
            x1_spanning_header_lines,
            y1_spanning_header_lines,
        ) = ([] for _ in range(17))

        image = get_image(file_path)
        img_width, img_height = get_image_size(file_path)

        (
            page_table_rotated,
            page_mix_ocr,
            page_rotation,
            page_searchable,
            objects,
            structure_transform,
        ) = process_page_characteristics(image, model, device, page)

        if (
            page_mix_ocr
            or page_rotation != 0
            or not page_searchable
            or page_table_rotated
        ):
            logging.info(
                f"Page {page_num} has a rotated or OCR-mixed table. Processing accordingly."
            )
            try:
                tables_crops = objects_to_crops(
                    image,
                    tokens,
                    objects,
                    detection_class_thresholds,
                    padding=crop_padding,
                )
                for i in range(len(tables_crops)):
                    maxi = i
                    (
                        new_df,
                        updated_df,
                        mapped_df,
                        row_lines,
                        row_headers,
                        pdf_column_header_matrix,
                    ) = process_table_image(
                        tables_crops,
                        i,
                        device,
                        structure_transform,
                        structure_model,
                        outputs_to_objects,
                        ocr,
                        crop_padding,
                        data_cropped_path,
                        objects,
                    )
                    table_type = identify_table_type(mapped_df)
                    if table_type == "Numeric":
                        logging.info(
                            f"Numeric table found on page {page_num}, table {i}."
                        )
                        image_path = f"{data_cropped_path}/cropped_images/cropped_table_image.png"
                        new_main_df = process_numeric_table(
                            image_path, pdf_column_header_matrix, row_lines, mapped_df
                        )
                        df_to_pickle(
                            new_main_df,
                            temp_output_folder,
                            page_num,
                            maxi,
                            rotated="yes",
                        )
                    else:
                        logging.info(
                            f"Non-numeric table found on page {page_num}, table {i}."
                        )
                        image_path = f"{data_cropped_path}/cropped_images/cropped_table_image.png"
                        process_non_numeric_table(
                            image_path,
                            pdf_column_header_matrix,
                            row_lines,
                            new_df,
                            mapped_df,
                            row_headers,
                            temp_output_folder,
                            page_num,
                            maxi,
                            rotated="yes",
                        )
            except Exception as e:
                logging.error(
                    f"Detected Rotated Table not processed on page {page_num}: {e}"
                )

        elif objects:
            if objects[0]["label"] == "table":
                logging.info(f"Detected table on page {page_num}.")
                try:
                    tables_crops = objects_to_crops(
                        image,
                        tokens,
                        objects,
                        detection_class_thresholds,
                        padding=crop_padding,
                    )
                    for i in range(len(tables_crops)):
                        maxi = i
                        (
                            new_df,
                            updated_df,
                            mapped_df,
                            row_lines,
                            row_headers,
                            pdf_column_header_matrix,
                        ) = process_table_image(
                            tables_crops,
                            i,
                            device,
                            structure_transform,
                            structure_model,
                            outputs_to_objects,
                            ocr,
                            crop_padding,
                            data_cropped_path,
                            objects,
                        )
                        table_type = identify_table_type(mapped_df)
                        if table_type == "Numeric":
                            logging.info(
                                f"Numeric table found on page {page_num}, table {i}."
                            )
                            image_path = f"{data_cropped_path}/cropped_images/cropped_table_image.png"
                            new_main_df = process_numeric_table(
                                image_path,
                                pdf_column_header_matrix,
                                row_lines,
                                mapped_df,
                            )
                            df_to_pickle(
                                new_main_df,
                                temp_output_folder,
                                page_num,
                                maxi,
                                rotated="no",
                            )
                        else:
                            logging.info(
                                f"Non-numeric table found on page {page_num}, table {i}."
                            )
                            image_path = f"{data_cropped_path}/cropped_images/cropped_table_image.png"
                            process_non_numeric_table(
                                image_path,
                                pdf_column_header_matrix,
                                row_lines,
                                new_df,
                                mapped_df,
                                row_headers,
                                temp_output_folder,
                                page_num,
                                maxi,
                                rotated="no",
                            )
                except Exception as e:
                    logging.error(
                        f"Detected Table not processed on page {page_num}: {e}"
                    )
            else:
                logging.info(f"No table found on page {page_num}.")
        else:
            logging.info(f"No objects detected on page {page_num}.")
