# Standard Libraries
import os
import re
from collections import defaultdict

# Data Processing
import pandas as pd

# PDF Processing and OCR
import fitz  # PyMuPDF
from fitz import Rect

# Image Processing
from PIL import Image
import cv2

# Machine Learning and Transformers
import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection

# Hugging Face and OpenAI Integrations
import openai


def initialize_table_models():
    """
    Initializes and loads the Table Transformer Detection and Table Structure Recognition models onto the available device (CPU or GPU).

    Returns:
    tuple: A tuple containing the initialized object detection model and the table structure recognition model.
    """
    # Determine the device to use: GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Table Transformer Detection model
    detection_model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    detection_model.to(device)

    # Load the Table Structure Recognition model
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )
    structure_model.to(device)

    # Return both models
    return detection_model, structure_model


def create_output_folder(output_folder_name):
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


def find_pages_with_keywords(doc, keywords):
    """
    Find the page numbers in a document where any of the specified keywords are present.

    Parameters:
    doc (Document): The document object to search through.
    keywords (list): A list of keywords to search for in the document.

    Returns:
    list: A list of page numbers (0-based) where the keywords are found.
    """
    pages_with_keywords = []

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the page
        text = page.get_text("text")  # Extract text from the page

        # Check if any of the keywords are present in the page text
        if any(keyword in text for keyword in keywords):
            pages_with_keywords.append(page_num)  # Append the 0-based page number

    return pages_with_keywords


def extract_blocks_with_coords(page):
    text_blocks, coords = [], []
    for word in page.get_text("words"):
        coords.append((word[0], word[1], word[2], word[3]))
        text_blocks.append(word[4])
    return text_blocks, coords


def find_near_duplicates(sequence, diff=5):
    updated_tally = defaultdict(list)
    for index, item in enumerate(sequence):
        updated_tally[round(item / diff) * diff].append(index)
    return updated_tally


def remove_y_duplicates(lines, threshold=2):
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
        len([d for d in average_distance if d > 8]) / len(average_distance) > 0.10
        or perc_vertical >= 0.8 * perc_horizontal
        and count_lines >= 3
        or numeric_content_count / len(text_blocks) > 0.15
    )


def find_pages_with_tables(doc):
    """
    Detect pages that likely contain tables based on text direction, block analysis, and line detection.

    Parameters:
    doc (Document): The document object to analyze.

    Returns:
    list: A list of page numbers (0-based) that likely contain tables.
    """
    pages_tables_contain = []

    for page_num, page in enumerate(doc):
        # Extract text blocks and their coordinates
        text_blocks, coords = extract_blocks_with_coords(page)
        y_coords = [coord[1] for coord in coords]

        # Update tally for near-duplicate coordinates
        find_near_duplicates(y_coords)

        # Clean lines by removing near-duplicate coordinates
        cleaned_lines = remove_y_duplicates(
            [(coord[:2], coord[2:]) for coord in coords]
        )

        # Count lines based on a certain threshold
        count_lines = len(
            [
                line
                for line in cleaned_lines
                if abs(line[0][0] - line[1][0]) > 120
                or abs(line[0][1] - line[1][1]) > 120
            ]
        )

        # Analyze text direction to help detect tables
        count_vertical, count_horizontal = 0, 0
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if line["dir"][0] == 0:  # Vertical lines
                        count_vertical += 1
                    else:  # Horizontal lines
                        count_horizontal += 1

        # Calculate the percentage of vertical and horizontal lines
        perc_vertical = count_vertical / (count_vertical + count_horizontal + 0.01)
        perc_horizontal = count_horizontal / (count_vertical + count_horizontal + 0.01)

        # Detect if the page contains a table based on the analyzed parameters
        contains_table = detect_table(
            page, text_blocks, count_lines, perc_vertical, perc_horizontal
        )
        if contains_table:
            pages_tables_contain.append(
                page_num
            )  # Append 0-based page number if a table is detected

    return pages_tables_contain


def save_pdf_pages_as_images(pdf_path, page_numbers, output_folder):
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
        # print(f"Page {page_number} saved as {output_path}")

    # Close the PDF document
    doc.close()


def get_image(file_path):
    image = Image.open(file_path).convert("RGB")
    # let's display it a bit smaller
    width, height = image.size
    image.resize((int(0.6 * width), (int(0.6 * height))))
    return image


def get_image_size(file_path):
    image = Image.open(file_path).convert("RGB")
    return image.size


def get_transform():
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
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    # print(pixel_values.shape)
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    with torch.no_grad():
        outputs = model(pixel_values)
    objects = outputs_to_objects(outputs, image.size, id2label)
    return objects


class MaxResize(object):
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


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def objects_to_crops(img, tokens, objects, class_thresholds, padding=0):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
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
        print(table_tokens)
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


def process_and_save_cropped_image(rgb_cropped_image, output_folder="data/RCT"):
    """
    Saves a cropped image, reads it back, applies a threshold, and saves the thresholded image.
    All operations are done within a directory named 'cropped_images' inside the specified output folder.

    :param rgb_cropped_image: A PIL Image object representing the cropped image to process.
    :param output_folder: The base directory where the 'cropped_images' folder will be created.
    """
    # Create the base directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the 'cropped_images' subdirectory
    cropped_images_folder = os.path.join(output_folder, "cropped_images")

    # Create the 'cropped_images' folder if it doesn't exist
    if not os.path.exists(cropped_images_folder):
        os.makedirs(cropped_images_folder)

    # Define file paths for saving the images
    cropped_img_path = os.path.join(cropped_images_folder, "cropped_table_image.png")
    thresholded_img_path = os.path.join(cropped_images_folder, "thresholded_image.png")

    # Save the cropped image
    rgb_cropped_image.save(cropped_img_path)

    # Read the image back using OpenCV
    img1 = cv2.imread(cropped_img_path)

    # Apply threshold
    _, thresh = cv2.threshold(img1, 170, 255, cv2.THRESH_BINARY)

    # Save the thresholded image
    cv2.imwrite(thresholded_img_path, thresh)
    return thresholded_img_path


def search_rct_in_area(pdf_document, page_number, y0, y1):
    # Open the PDF file
    doc = fitz.open(pdf_document)

    # Load the specified page (convert to 0-based index)
    page = doc.load_page(page_number)
    blocks = page.get_text("blocks")  # Extract text blocks

    # Iterate through each block
    for block in blocks:
        bbox = block[:4]  # Bounding box coordinates (x0, y0, x1, y1)
        text = block[4]  # Extracted text

        # Check if the block is within the specified vertical area
        if y0 <= bbox[1] <= y1 or y0 <= bbox[3] <= y1:
            # Check if "RCT" is present in the text
            if "RCT" in text:
                return "yes"

    return "no"


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
    integer_list = [int(ly) for ly in y0ss]

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


# Function to classify table headers from a LaTeX table using GPT-4
def classify_headers_from_latex_with_gpt(latex_content):
    # Preprocess the LaTeX content to get only the table part
    table_pattern = re.compile(r"\\begin{tabular}.*?\\end{tabular}", re.DOTALL)
    table_match = table_pattern.search(latex_content)

    # If no table found, raise an error
    if not table_match:
        raise ValueError("No table found in the LaTeX content")

    table_content = latex_content
    # Escape curly braces in the LaTeX content
    escaped_table_content = table_content.replace("{", "{{").replace("}", "}}")

    # Prepare the prompt for GPT-4
    prompt = f"""
    Given a LaTeX table entry describing a study, classify whether the study type is an RCT or not. Look for specific keywords and phrases that indicate randomization and controlled conditions such as Randomised controlled trial. The output should be "RCT" if the study is a Randomized Controlled Trial and "Not RCT" otherwise.
    Describe this result in the provided latex table using the JSON templates below. The author names might include organization names as well, for example "World Bank." Use the appropriate template based on the type of study: "RCT", "Not RCT". For any unanswerable attributes in the templates, set their value to the placeholder "xx" if it is a string type.
    If there are equal to or more than two authors present then write it in a separate json with properly aligned year. Don't include any stop words such as 'and'.


    JSON Templates:

    {{"author name": "xx", "type": "RCT", "year": "xx"}}
    {{"author name": "xx", "type": "Not RCT", "year": "xx"}}


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
                "content": "You are an assistant that extracts the author names as a list, as well as the year, and maps them to the corresponding study type, determining whether it is an RCT or not.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0,
    )

    # Extract the classifications from the response
    classifications = response["choices"][0]["message"]["content"].strip()
    return classifications
