import pandas as pd
import re
from fuzzywuzzy import fuzz
import json

from utils import (
    extract_word_positions,
    create_output_folder,
    save_pdf_pages_as_images,
    extract_text_and_captions,
    is_valid_caption,
    detect_figure_regions,
    calculate_distances,
    complete_captions,
    extract_image_blocks,
    find_lines_with_large_x_diff,
    find_nearest_captions,
    filter_regions,
    process_image_to_latex,
    process_regions,
    merge_words_into_lines,
    union_of_lists,
    consecutive_sublists,
    get_image_dimensions,
    get_pdf_dimensions,
    pdf_to_image_coordinates,
    find_unique_lines_column,
    find_unique_lines_row,
    classify_headers_from_latex_with_gpt,
    assign_row_col_numbers,
    get_pdf_page_dimensions,
    check_y_diff_within_threshold_text_classification,
    get_y0,
    crop_image,
    process_ocr_results,
    adjust_signs_to_match_rule,
    ImageClassifier,
)


# Predefined pattern
updated_pattern_with_conjunctions_and_next_word = r"(?:Table|Figure|figure|fig|Fig|Fig.|Analysis)\s+[A-Z]*\.*\d*[a-z]*(?:\s+(?:and|&|\?|\/|or|)\s+\w+)?"


# Function to extract table names including conjunctions and next word
def extract_tables_including_conjunctions_and_next_word(captions, pattern):
    extracted_names = []
    for caption in captions:
        matches = re.findall(pattern, caption, re.IGNORECASE)
        for match in matches:
            table_name = " ".join(match.split())
            extracted_names.append(table_name)
    return extracted_names


# Function to perform fuzzy filtering
def fuzzy_filter(
    df, pattern, initial_threshold=40, min_threshold=30, max_threshold=100
):
    threshold = initial_threshold
    results = df["block_text_lower"].apply(lambda x: fuzz.partial_ratio(x, pattern))

    while results[results >= threshold].empty and threshold > min_threshold:
        threshold -= 5
        results = df["block_text_lower"].apply(lambda x: fuzz.partial_ratio(x, pattern))

    while not results[results >= threshold].empty and threshold < max_threshold:
        threshold += 5
        results = df["block_text_lower"].apply(lambda x: fuzz.partial_ratio(x, pattern))

    filtered_df = df[results >= threshold]
    return filtered_df, results[results >= threshold]


# Main function to preprocess and extract captions
def new_extract_captions(df_extracted_rule_features, df_new):
    # Preprocessing
    df_extracted_rule_features["block_text_lower"] = (
        df_extracted_rule_features["text"]
        .str.lower()
        .str.strip()
        .replace(r"\s+", " ", regex=True)
    )
    df_new["caption_lowercase"] = (
        df_new["Caption"].str.lower().str.strip().replace(r"\s+", " ", regex=True)
    )
    unique_captions = df_new["caption_lowercase"].unique()

    extracted_captions = []

    mini_count = 0
    for caption in unique_captions:
        # Extract names using the provided pattern
        extracted_names = extract_tables_including_conjunctions_and_next_word(
            [caption], updated_pattern_with_conjunctions_and_next_word
        )

        if len(extracted_names) == 0:
            continue

        # Get the length of the extracted names
        mention_length = len(extracted_names[0])

        # Filter the dataframe based on the extracted names
        pattern = caption[: mention_length + 2]
        filtered_caption, results = fuzzy_filter(df_extracted_rule_features, pattern)

        # If there are multiple entries, pick the one with the highest index
        if filtered_caption.shape[0] == 0:
            x1 = df_new.loc[df_new["caption_lowercase"] == caption, "BBox"].values[0][2]
            y1 = df_new.loc[df_new["caption_lowercase"] == caption, "BBox"].values[0][3]
            x0 = df_new.loc[df_new["caption_lowercase"] == caption, "BBox"].values[0][0]
            y0 = df_new.loc[df_new["caption_lowercase"] == caption, "BBox"].values[0][1]
            page = df_new.loc[df_new["caption_lowercase"] == caption, "Page"].values[0]
            extracted_captions.append(
                {
                    "original_caption": caption,
                    "extracted_text": caption,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "page": page,
                }
            )
        else:
            extracted_text = filtered_caption["text"].values[-1]
            x1 = filtered_caption["true_x2"].values[-1]
            y1 = filtered_caption["true_y2"].values[-1]
            x0 = filtered_caption["true_x1"].values[-1]
            y0 = filtered_caption["true_y1"].values[-1]
            page = filtered_caption["#page"].values[-1]
            extracted_captions.append(
                {
                    "original_caption": caption,
                    "extracted_text": extracted_text,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "page": page,
                }
            )

        mini_count = mini_count + 1

    # Create a new DataFrame with the extracted captions

    df_extracted_captions = pd.DataFrame(extracted_captions)
    return df_extracted_captions


def process_extracted_captions(df_rule_features, df_new):
    # Merge or process the dataframes as per the original extraction function logic
    df_extracted_captions = new_extract_captions(df_rule_features, df_new)

    # Create BBox column by combining coordinates
    df_extracted_captions["BBox"] = df_extracted_captions[
        ["x0", "y0", "x1", "y1"]
    ].values.tolist()

    # Create y_mid and x_mid columns
    df_extracted_captions["y_mid"] = (
        df_extracted_captions["y0"] + df_extracted_captions["y1"]
    ) / 2
    df_extracted_captions["x_mid"] = (
        df_extracted_captions["x0"] + df_extracted_captions["x1"]
    ) / 2

    # Rename extracted_text and page columns
    df_extracted_captions["Caption"] = df_extracted_captions["extracted_text"]
    df_extracted_captions["Page"] = df_extracted_captions["page"]

    # Drop unnecessary columns
    df_extracted_captions.drop(["extracted_text", "page"], axis=1, inplace=True)

    # Remove duplicates based on y_mid, x_mid, Page, and Caption
    df_extracted_captions = df_extracted_captions.drop_duplicates(
        subset=["y_mid", "x_mid", "Page", "Caption"]
    )

    # Sort by Page and y_mid
    df_extracted_captions.sort_values(by=["Page", "y_mid"], inplace=True)

    return df_extracted_captions


def process_and_save_images(
    page_numbers, pdf_path, output_folder_path, output_folder_name="Images"
):
    # Create a copy of the dataframe

    # Create output folder for images
    output_folder_image_path = create_output_folder(
        output_folder_path, output_folder_name
    )

    # Convert specified pages to images and save
    for page in page_numbers:
        page_numbers_to_convert = [int(page)]
        save_pdf_pages_as_images(
            pdf_path, page_numbers_to_convert, output_folder_image_path
        )


def preprocess_document_and_features(df_extracted_rule_features, df_new):
    # Preprocessing: Convert text columns to lowercase, strip whitespaces, and replace multiple spaces
    df_extracted_rule_features["block_text_lower"] = (
        df_extracted_rule_features["text"]
        .str.lower()
        .str.strip()
        .replace(r"\s+", " ", regex=True)
    )

    df_new["caption_lowercase"] = (
        df_new["Caption"].str.lower().str.strip().replace(r"\s+", " ", regex=True)
    )
    return df_extracted_rule_features, df_new


def extract_and_process_captions(pdf_path):
    # Extract text data and captions from the PDF
    text_data, captions = extract_text_and_captions(pdf_path)

    # Create a DataFrame with the extracted captions
    df = pd.DataFrame(captions, columns=["Page", "Caption"])

    # Validate captions using the is_valid_caption function
    df["valid"] = df["Caption"].apply(is_valid_caption)

    # Filter valid captions
    df_new = df[df["valid"] == "Yes"]

    # Complete captions processing
    df_new = complete_captions(df_new, pdf_path)

    return df_new


def process_page(number_page, pdf_path, df_new, api_key, ocr):
    """
    Process a single page of the PDF to extract regions, detect figures, and classify images.
    """
    df_words, page_widths = extract_word_positions(pdf_path, page_num=number_page)
    df_lines, flagged_lines = merge_words_into_lines(df_words, page_widths)
    distances_df = calculate_distances(df_lines)
    text_lines = find_lines_with_large_x_diff(distances_df)
    result = union_of_lists(flagged_lines, list(text_lines))
    result.sort()
    result_sublists = consecutive_sublists(result)

    # Filter result sublists
    result_sublists = filter_sublists_by_y_diff(df_lines, result_sublists, threshold=55)

    inbuilt_regions, xy_regions = extract_image_blocks(pdf_path, page_num=number_page)
    figure_regions = detect_figure_regions(pdf_path, page_num=number_page)
    range_header, range_footnote = get_header_and_footer_ranges(pdf_path, number_page)

    rule_filtered_regions = filter_regions(figure_regions, range_header, range_footnote)
    new_inbuilt_regions = filter_regions(inbuilt_regions, range_header, range_footnote)

    ocr_bbox = process_regions(rule_filtered_regions, new_inbuilt_regions)
    result_sublist_bbox = get_y0_bounding_boxes(df_lines, result_sublists)
    new_result_sublist_bbox = filter_regions(
        result_sublist_bbox, range_header, range_footnote
    )
    main_bbox = sorted(new_result_sublist_bbox + ocr_bbox, key=lambda x: x[0])

    lo_cap = df_new[df_new["Page"] == number_page]
    if len(main_bbox) >= len(lo_cap):
        caption_text, total = process_figures_with_more_or_equal_captions(
            main_bbox,
            inbuilt_regions,
            lo_cap,
            page_widths,
            pdf_path,
            number_page,
            api_key,
            xy_regions,
            ocr,
        )
    else:
        caption_text, total = process_figures_with_fewer_captions(
            main_bbox,
            inbuilt_regions,
            lo_cap,
            page_widths,
            pdf_path,
            number_page,
            api_key,
            xy_regions,
            ocr,
        )
    return caption_text, total


def get_header_and_footer_ranges(pdf_path, number_page):
    """
    Calculate header and footer ranges for a given page based on its height.
    """
    width, height = get_pdf_page_dimensions(pdf_path, page_number=number_page)
    footnote_start = height - height / 10
    range_footnote = [footnote_start, height]
    range_header = [0, height / 10]
    return range_header, range_footnote


def filter_sublists_by_y_diff(df_lines, result_sublists, threshold):
    """
    Filter sublists by checking y-difference within threshold using text classification.
    """
    for sublist in result_sublists:
        result = check_y_diff_within_threshold_text_classification(
            df_lines, sublist, threshold=threshold
        )
        if not result:
            result_sublists.remove(sublist)
    return result_sublists


def filter_figure_and_inbuilt_regions(
    figure_regions, inbuilt_regions, range_header, range_footnote
):
    """
    Filter figure and inbuilt regions based on header and footer ranges.
    """
    rule_filtered_regions = filter_regions(figure_regions, range_header, range_footnote)
    new_inbuilt_regions = filter_regions(inbuilt_regions, range_header, range_footnote)
    return rule_filtered_regions, new_inbuilt_regions


def get_y0_bounding_boxes(df_lines, result_sublists):
    """
    Get bounding boxes based on y0 values from result sublists.
    """
    result_sublist_bbox = []
    for sublist in result_sublists:
        result = get_y0(df_lines, sublist)
        result_sublist_bbox.append(result)
    return result_sublist_bbox


def process_figures_with_more_or_equal_captions(
    main_bbox,
    inbuilt_regions,
    lo_cap,
    page_widths,
    pdf_path,
    number_page,
    api_key,
    xy_regions,
    ocr,
):
    count = 0
    main_bbox.sort(key=lambda x: x[0], reverse=True)
    inbuilt_regions.sort(key=lambda x: x[0], reverse=True)
    sorted_captions = lo_cap.sort_values(by="y_mid", ascending=False)
    caption_list = sorted_captions["Caption"].tolist()

    for region in main_bbox:
        x0, y0, x1, y1 = get_coordinates_for_region(
            region, inbuilt_regions, xy_regions, page_widths
        )
        saved_image_path = save_cropped_image(
            pdf_path, x0, y0, x1, y1, number_page, count
        )
        caption_text, total = classify_image_and_process(
            caption_list,
            count,
            api_key,
            number_page,
            region,
            inbuilt_regions,
            saved_image_path,
            ocr,
        )

        # Handle cases where the function returns None or unexpected types
        if caption_text is None:
            caption_text = f"Default_Caption_{count}_{number_page + 1}"
        if total is None:
            total = []

        return caption_text, total


def process_figures_with_fewer_captions(
    main_bbox,
    inbuilt_regions,
    lo_cap,
    page_widths,
    pdf_path,
    number_page,
    api_key,
    xy_regions,
    ocr,
):
    count = 0
    main_bbox.sort(key=lambda x: x[0], reverse=False)
    inbuilt_regions.sort(key=lambda x: x[0], reverse=False)
    sorted_captions = lo_cap.sort_values(by="y_mid", ascending=True)
    caption_list = sorted_captions["Caption"].tolist()

    for region in main_bbox:
        x0, y0, x1, y1 = get_coordinates_for_region(
            region, inbuilt_regions, xy_regions, page_widths
        )
        saved_image_path = save_cropped_image(
            pdf_path, x0, y0, x1, y1, number_page, count
        )
        caption_text, total = classify_image_and_process(
            caption_list,
            count,
            api_key,
            number_page,
            region,
            inbuilt_regions,
            saved_image_path,
            ocr,
        )

        # Handle cases where the function returns None or unexpected types
        if caption_text is None:
            caption_text = f"Default_Caption_{count}_{number_page + 1}"
        if total is None:
            total = []
        print(caption_text)

        return caption_text, total


def get_coordinates_for_region(region, inbuilt_regions, xy_regions, page_widths):
    """
    Get coordinates for cropping an image based on region position.
    """
    if region in inbuilt_regions:
        match_found = False
        for region_check in xy_regions:
            if region[0] == region_check[1] and region[1] == region_check[3]:
                x0, y0, x1, y1 = region_check[0], region[0], region_check[2], region[1]
                match_found = True
                break
        if not match_found:
            x0, y0, x1, y1 = (
                page_widths[0] * 0.04,
                region[0],
                page_widths[0] * 0.96,
                region[1],
            )
    else:
        x0, y0, x1, y1 = (
            page_widths[0] * 0.04,
            region[0],
            page_widths[0] * 0.96,
            region[1],
        )
    return x0, y0, x1, y1


def save_cropped_image(pdf_path, x0, y0, x1, y1, number_page, count):
    """
    Save the cropped image from the extracted coordinates.
    """
    pdf_width, pdf_height = get_pdf_dimensions(pdf_path)
    img_width, img_height = get_image_dimensions(
        f"data/ForestPlots/Images/page_{int(number_page)}.png"
    )
    x0, y0 = pdf_to_image_coordinates(
        x0, y0, pdf_width, pdf_height, img_width, img_height
    )
    x1, y1 = pdf_to_image_coordinates(
        x1, y1, pdf_width, pdf_height, img_width, img_height
    )

    image_path = f"data/ForestPlots/Images/page_{number_page}.png"
    save_folder = "data/ForestPlots/cropped_image"
    saved_image_path = crop_image(
        image_path,
        x0 - 30,
        y0 - 30,
        x1 + 30,
        y1 + 60,
        save_folder,
        mention=[f"page_{count}"],
    )
    print("Cropped image saved to:", saved_image_path)
    return saved_image_path


def classify_image_and_process(
    caption_list,
    count,
    api_key,
    number_page,
    region,
    inbuilt_regions,
    saved_image_path,
    ocr,
):
    """
    Classify the image and process it if it's a Forest Plot.
    """
    image_path = f"data/ForestPlots/cropped_image/page_{count}.png"
    classifier = ImageClassifier(api_key, image_path)
    result = classifier.classify_image()

    if result["choices"][0]["message"]["content"] == "Forest Plot":
        latex_content = process_image_to_latex(
            saved_image_path,
            ocr,
            process_ocr_results,
            find_unique_lines_row,
            find_unique_lines_column,
            assign_row_col_numbers,
        )
        caption_text = (
            caption_list[count]
            if count < len(caption_list)
            else f"No_Caption_{count}_{number_page + 1}"
        )
        classifications = classify_headers_from_latex_with_gpt(
            latex_content, caption_text
        )
        caption_text, total = extract_json_and_store(
            classifications, number_page, region, caption_text
        )
        print(caption_text)
        print(total)
        return caption_text, total
    else:
        return None, []


def extract_json_and_store(classifications, number_page, region, caption_text):
    """
    Extract JSON objects from classifications and store them in a dictionary.
    """
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, classifications, re.DOTALL)
    total = []

    for match_i in matches:
        try:
            json_data = json.loads(match_i)
            json_data["page_number"] = number_page + 1
            json_data["y_sorting"] = (region[0] + region[1]) / 2
            total.append(json_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)

    # Dictionary[f"{caption_text}"] = total

    return caption_text, total


def convert_dictionary_to_dataframe(data):
    """
    Convert the data dictionary into a DataFrame with appropriate handling for missing or empty data.
    """
    dfs = []
    # Iterate over each caption and records to create DataFrames
    for caption, records in data.items():
        if records:  # Check if records are not empty
            df = pd.DataFrame(records)  # Convert the list of dicts to a DataFrame
            df["Caption"] = caption  # Add a column for the figure name
            dfs.append(df)
        else:
            print(f"No records found for caption: {caption}")

    # Concatenate all DataFrames, handle empty list scenario
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    else:
        print("No DataFrames to concatenate.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid data


def adjust_dataframe_values(df):
    """
    Adjust signs of values in the DataFrame based on custom rules.
    """
    # Iterate over each row in the DataFrame using a for loop
    for index, row in df.iterrows():
        try:
            value = float(row["value"])
            ci_low = float(row["confidence interval"][0])
            ci_high = float(row["confidence interval"][1])

            adjusted_values = adjust_signs_to_match_rule(value, ci_low, ci_high)

            adjusted_x, adjusted_y, adjusted_z = adjusted_values
            # Update the DataFrame with adjusted values
            df.at[index, "value"] = adjusted_x
            df.at[index, "confidence interval"] = [adjusted_y, adjusted_z]

        except Exception as e:
            print(f"Skipping processing row for value check {index}: {e}")

    return df


def sort_dataframe(df, sort_columns):
    """
    Sort the DataFrame based on specified columns.
    """
    if not df.empty:
        df_sorted = df.sort_values(by=sort_columns)
        return df_sorted
    else:
        print("The DataFrame is empty, skipping sorting.")
        return df


def extract_string_from_path(path):
    """
    Extract the string component from a file path.
    """
    try:
        # Split the path to get the last component after the last '/'
        file_name = path.split("/")[-1]
        # Split the file name at the first '.' to extract the part before the file extension
        extracted_string = file_name.split(".")[0]
        return extracted_string
    except Exception as e:
        print(f"Error extracting string from path: {e}")
        return ""


# Example usage
def process_data(Dictionary, pdf_path):
    """
    Main function to process the data from the Dictionary and handle the entire workflow.
    """
    # Convert the Dictionary into a DataFrame
    full_df = convert_dictionary_to_dataframe(Dictionary)

    if not full_df.empty:
        # Copy and adjust values in the DataFrame
        adjusted_df = adjust_dataframe_values(full_df.copy())

        # Sort the DataFrame by 'page_number' and 'y_sorting'
        df_sorted = sort_dataframe(adjusted_df, ["page_number", "y_sorting"])
    else:
        print("No data to process.")
        df_sorted = pd.DataFrame()

    # Extract string from the given pdf path
    extracted_string = extract_string_from_path(pdf_path)

    return df_sorted, extracted_string


def main_Forest(
    page_numbers, pdf_path, df_new, api_key, Dictionary, ocr, data_output_path
):
    for number_page in page_numbers:
        number_page = int(number_page)
        print("-" * 50)
        print(f"Extracting FPs for page number: {number_page}")
        print("-" * 50)

        df_words, page_widths = extract_word_positions(pdf_path, page_num=number_page)

        df_lines, flagged_lines = merge_words_into_lines(df_words, page_widths)

        # Calculate distances
        distances_df = calculate_distances(df_lines)
        text_lines = find_lines_with_large_x_diff(distances_df)

        result = union_of_lists(flagged_lines, list(text_lines))
        result.sort()

        result_sublists = consecutive_sublists(result)

        for sublist in result_sublists:
            result = check_y_diff_within_threshold_text_classification(
                df_lines, sublist, threshold=55
            )
            if result is False:
                del result_sublists[result_sublists.index(sublist)]

        inbuilt_regions, xy_regions = extract_image_blocks(
            pdf_path, page_num=number_page
        )

        figure_regions = detect_figure_regions(pdf_path, page_num=number_page)

        width, height = get_pdf_page_dimensions(pdf_path, page_number=number_page)
        footnote_start = height - height / 10
        footnote_end = height
        range_footnote = [footnote_start, footnote_end]
        range_header = [0, height / 10]

        rule_filtered_regions = filter_regions(
            figure_regions, range_header, range_footnote
        )
        new_inbuilt_regions = filter_regions(
            inbuilt_regions, range_header, range_footnote
        )

        ocr_bbox = process_regions(rule_filtered_regions, new_inbuilt_regions)

        result_sublist_bbox = []
        for sublist in result_sublists:
            result = get_y0(df_lines, sublist)
            result_sublist_bbox.append(result)

        new_result_sublist_bbox = filter_regions(
            result_sublist_bbox, range_header, range_footnote
        )

        main_bbox = new_result_sublist_bbox + ocr_bbox
        main_bbox = sorted(main_bbox, key=lambda x: x[0])
        inbuilt_regions = sorted(inbuilt_regions, key=lambda x: x[0])

        lo_cap = df_new[df_new["Page"] == number_page]

        ### Number of Figures are higher or equal to the number of captions. Follow bottom up approach

        if len(main_bbox) == 0:
            pass

        elif len(main_bbox) >= len(lo_cap):
            count = 0
            main_bbox = sorted(main_bbox, key=lambda x: x[0], reverse=True)
            inbuilt_regions = sorted(inbuilt_regions, key=lambda x: x[0], reverse=True)
            # Sort the filtered DataFrame by the y_mid value
            sorted_captions = lo_cap.sort_values(by="y_mid", ascending=False)

            # Extract the captions into a list
            caption_list = sorted_captions["Caption"].tolist()
            cap = 0
            for region in main_bbox:
                if region in inbuilt_regions:
                    match_found = False
                    for region_check in xy_regions:
                        if (
                            region[0] == region_check[1]
                            and region[1] == region_check[3]
                        ):
                            x0, y0, x1, y1 = (
                                region_check[0],
                                region[0],
                                region_check[2],
                                region[1],
                            )
                            match_found = True
                            break

                        if not match_found:
                            x0, y0, x1, y1 = (
                                page_widths[0] * 0.04,
                                region[0],
                                page_widths[0] * 0.96,
                                region[1],
                            )
                else:
                    x0, y0, x1, y1 = (
                        page_widths[0] * 0.04,
                        region[0],
                        page_widths[0] * 0.96,
                        region[1],
                    )

                figure_bboxes = [
                    {"id": f"Figure {i+1}", "y_start": min(bbox), "y_end": max(bbox)}
                    for i, bbox in enumerate(main_bbox)
                ]
                nearest_captions = find_nearest_captions(sorted_captions, figure_bboxes)

                # print(x0, y0, x1, y1)

                pdf_width, pdf_height = get_pdf_dimensions(pdf_path)
                img_width, img_height = get_image_dimensions(
                    f"{data_output_path}/Images/page_{int(number_page)}.png"
                )
                x0, y0 = pdf_to_image_coordinates(
                    x0, y0, pdf_width, pdf_height, img_width, img_height
                )
                x1, y1 = pdf_to_image_coordinates(
                    x1, y1, pdf_width, pdf_height, img_width, img_height
                )
                print("second")
                print(x0, y0, x1, y1)

                image_path = f"{data_output_path}/Images/page_{number_page}.png"

                save_folder = f"{data_output_path}/cropped_image"
                saved_image_path = crop_image(
                    image_path,
                    x0,
                    y0 - 30,
                    x1 + 30,
                    y1 + 60,
                    save_folder,
                    mention=[f"page_{count}"],
                )
                print("Cropped image saved to:", saved_image_path)
                image_path = f"{data_output_path}/cropped_image/page_{count}.png"

                count = count + 1

                classifier = ImageClassifier(api_key, image_path)
                result = classifier.classify_image()
                try:
                    if result["choices"][0]["message"]["content"] == "Forest Plot":
                        latex_content = process_image_to_latex(
                            saved_image_path,
                            ocr,
                            process_ocr_results,
                            find_unique_lines_row,
                            find_unique_lines_column,
                            assign_row_col_numbers,
                        )
                        print("***********************************************")
                        print(cap)
                        print("***********************************************")
                        print(len(caption_list))
                        try:
                            caption_text = caption_list[cap]
                            caption_text = nearest_captions[f"Figure {count}"]
                        except IndexError:
                            caption_text = f"No_Caption_{cap}_{number_page + 1}"
                        print(caption_text)

                        classifications = classify_headers_from_latex_with_gpt(
                            latex_content, caption_text
                        )
                        print(classifications)
                        cap = cap + 1
                        # Regex to match the JSON-like objects
                        pattern = r"\{.*?\}"
                        matches = re.findall(pattern, classifications, re.DOTALL)

                        # Print the matches
                        total = []
                        for match_i in matches:
                            # Optionally convert to dictionary using json.loads() if valid JSON
                            try:
                                json_data = json.loads(match_i)
                                json_data["page_number"] = number_page + 1
                                json_data["y_sorting"] = (region[0] + region[1]) / 2
                                total.append(json_data)
                            except json.JSONDecodeError as e:
                                print("Invalid JSON:", e)

                        Dictionary[f"{caption_text}"] = total
                    else:
                        print("Not a Forest plot")
                except:
                    pass

        else:
            count = 0
            main_bbox = sorted(main_bbox, key=lambda x: x[0], reverse=False)
            inbuilt_regions = sorted(inbuilt_regions, key=lambda x: x[0], reverse=False)
            # Sort the filtered DataFrame by the y_mid value
            sorted_captions = lo_cap.sort_values(by="y_mid", ascending=True)

            # Extract the captions into a list
            caption_list = sorted_captions["Caption"].tolist()

            figure_bboxes = [
                {"id": f"Figure {i+1}", "y_start": min(bbox), "y_end": max(bbox)}
                for i, bbox in enumerate(main_bbox)
            ]

            nearest_captions = find_nearest_captions(sorted_captions, figure_bboxes)

            for region in main_bbox:
                if region in inbuilt_regions:
                    match_found = False
                    for region_check in xy_regions:
                        if (
                            region[0] == region_check[1]
                            and region[1] == region_check[3]
                        ):
                            x0, y0, x1, y1 = (
                                region_check[0],
                                region[0],
                                region_check[2],
                                region[1],
                            )
                            match_found = True
                            break
                        if not match_found:
                            x0, y0, x1, y1 = (
                                page_widths[0] * 0.04,
                                region[0],
                                page_widths[0] * 0.96,
                                region[1],
                            )
                else:
                    x0, y0, x1, y1 = (
                        page_widths[0] * 0.04,
                        region[0],
                        page_widths[0] * 0.96,
                        region[1],
                    )

                pdf_width, pdf_height = get_pdf_dimensions(pdf_path)
                img_width, img_height = get_image_dimensions(
                    f"{data_output_path}/Images/page_{int(number_page)}.png"
                )
                x0, y0 = pdf_to_image_coordinates(
                    x0, y0, pdf_width, pdf_height, img_width, img_height
                )
                x1, y1 = pdf_to_image_coordinates(
                    x1, y1, pdf_width, pdf_height, img_width, img_height
                )

                image_path = f"{data_output_path}/Images/page_{number_page}.png"

                save_folder = f"{data_output_path}/cropped_image"
                saved_image_path = crop_image(
                    image_path,
                    x0 - 30,
                    y0 - 30,
                    x1 + 30,
                    y1 + 60,
                    save_folder,
                    mention=[f"page_{count}"],
                )
                print("Cropped image saved to:", saved_image_path)
                image_path = f"{data_output_path}/cropped_image/page_{count}.png"

                count = count + 1

                classifier = ImageClassifier(api_key, image_path)
                result = classifier.classify_image()
                if result["choices"][0]["message"]["content"] == "Forest Plot":
                    latex_content = process_image_to_latex(
                        saved_image_path,
                        ocr,
                        process_ocr_results,
                        find_unique_lines_row,
                        find_unique_lines_column,
                        assign_row_col_numbers,
                    )
                    caption_text = nearest_captions[f"Figure {count}"]
                    classifications = classify_headers_from_latex_with_gpt(
                        latex_content, caption_text
                    )

                    # Regex to match the JSON-like objects
                    pattern = r"\{.*?\}"
                    matches = re.findall(pattern, classifications, re.DOTALL)

                    # Print the matches
                    total = []
                    for match_i in matches:
                        # Optionally convert to dictionary using json.loads() if valid JSON
                        try:
                            json_data = json.loads(match_i)
                            json_data["page_number"] = number_page + 1
                            json_data["y_sorting"] = (region[0] + region[1]) / 2
                            total.append(json_data)
                        except json.JSONDecodeError as e:
                            print("Invalid JSON:", e)

                    Dictionary[f"{caption_text}"] = total

    return Dictionary
