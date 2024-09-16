# Standard library imports
import logging
import pickle
import re
from operator import itemgetter
from os.path import join


# Third-party imports
import fitz  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from fitz.utils import getColor  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
import collections


# it is important that each bounding box should be in (upper left, lower right) format.
# source: https://github.com/NielsRogge/Transformers-Tutorials/issues/129


def upperleft_to_lowerright(bbox: list) -> list:
    """
    Ensures that the bounding box coordinates are ordered correctly, with the upper-left corner first and the lower-right corner second.

    :param bbox: List of bounding box coordinates in the format [x0, y0, x1, y1].
    :return: List of bounding box coordinates in the format [x0, y0, x1, y1], with x0 <= x1 and y0 <= y1.
    """

    x0, y0, x1, y1 = bbox
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [x0, y0, x1, y1]


def convert_box(bbox: list) -> list:
    """
    Converts bounding box coordinates from (left, top, width, height) format to (left, top, left+width, top+height) format.

    :param bbox: List of bounding box coordinates in the format [left, top, width, height].
    :return: List of bounding box coordinates in the format [left, top, left+width, top+height].
    """

    x, y, w, h = tuple(bbox)
    return [x, y, x + w, y + h]


def normalize_box(bbox: list, width: int, height: int) -> list:
    """
    Normalizes bounding box coordinates for a 1000x1000 pixels image.

    :param bbox: List of bounding box coordinates.
    :param width: Width of the image.
    :param height: Height of the image.
    :return: List of normalized bounding box coordinates.
    """

    if width > 1000 or height > 1000:
        return [
            max(0, int(bbox[0] / width * 1000)),
            max(0, int(bbox[1] / height * 1000)),
            max(0, int(bbox[2] / width * 1000)),
            max(0, int(bbox[3] / height * 1000)),
        ]
    else:
        return [
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        ]


# LiLT model gets 1000x10000 pixels images
def denormalize_box(bbox: list, width: int, height: int) -> list:
    """
    Denormalizes bounding box coordinates for a 1000x1000 pixels image.

    :param bbox: List of normalized bounding box coordinates.
    :param width: Width of the image.
    :param height: Height of the image.
    :return: List of denormalized bounding box coordinates.
    """
    if width > 1000 or height > 1000:
        return [
            int(width * (bbox[0] / 1000)),
            int(height * (bbox[1] / 1000)),
            int(width * (bbox[2] / 1000)),
            int(height * (bbox[3] / 1000)),
        ]
    else:
        return bbox


def get_sorted_boxes(bboxes: list) -> list:
    """
    Sorts bounding boxes first by y-coordinate from top to bottom, then by x-coordinate from left to right for boxes with the same y-coordinate.

    :param bboxes: List of bounding boxes.
    :return: List of sorted bounding boxes.
    """

    # sort by y from page top to bottom
    sorted_bboxes = sorted(bboxes, key=itemgetter(1), reverse=False)
    y_list = [bbox[1] for bbox in sorted_bboxes]

    # sort by x from page left to right when boxes with same y
    if len(set(y_list)) != len(y_list):
        y_list_duplicates_indexes = dict()
        y_list_duplicates = [
            item for item, count in collections.Counter(y_list).items() if count > 1
        ]
        for item in y_list_duplicates:
            y_list_duplicates_indexes[item] = [
                i for i, e in enumerate(y_list) if e == item
            ]
            bbox_list_y_duplicates = sorted(
                np.array(sorted_bboxes, dtype=object)[
                    y_list_duplicates_indexes[item]
                ].tolist(),
                key=itemgetter(0),
                reverse=False,
            )
            np_array_bboxes = np.array(sorted_bboxes)
            np_array_bboxes[y_list_duplicates_indexes[item]] = np.array(
                bbox_list_y_duplicates
            )
            sorted_bboxes = np_array_bboxes.tolist()

    return sorted_bboxes


def sort_data_labels(bboxes: list, texts: list, labels=None) -> tuple:
    """
    Sorts data from y = 0 to end of page (and after, x=0 to end of page when necessary) with or without labels.

    :param bboxes: List of bounding boxes.
    :param texts: List of texts.
    :param labels: List of labels. Default is None.
    :return: Tuple of lists of sorted bounding boxes and texts. If labels are provided, also returns a list of sorted labels.
    """

    sorted_bboxes = get_sorted_boxes(bboxes)
    sorted_bboxes_indexes = [bboxes.index(bbox) for bbox in sorted_bboxes]
    sorted_texts = np.array(texts, dtype=object)[sorted_bboxes_indexes].tolist()
    if labels:
        sorted_labels = np.array(labels, dtype=object)[sorted_bboxes_indexes].tolist()
    else:
        sorted_labels = None

    return sorted_bboxes, sorted_texts, sorted_labels


class CustomDataset(Dataset):
    """
    Custom Dataset class for PyTorch.

    :param dataset: The dataset to be used.
    :param tokenizer: The tokenizer to be used.
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the length of the dataset.

        :return: Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        :param idx: Index of the item.
        :return: Item at the specified index.
        """
        example = self.dataset[idx]
        encoding = dict()

        if "images_ids" in example:
            encoding["images_ids"] = example["images_ids"]
        if "chunk_ids" in example:
            encoding["chunk_ids"] = example["chunk_ids"]
        if "labels" in example:
            encoding["labels"] = example["labels"]
        encoding["input_ids"] = example["input_ids"]
        encoding["attention_mask"] = example["attention_mask"]
        encoding["bbox"] = example["normalized_bboxes"]

        return encoding


def consolidate_blocks(
    example,
    keys=[
        "doc_category",
        "collection",
        "original_filename",
        "coco_width",
        "coco_height",
        "original_width",
        "original_height",
        "page_no",
        "num_pages",
    ],
):
    """
    Consolidate blocks of an example, grouping texts by their bounding box.

    :param example: The example containing 'bboxes_block', 'categories', 'texts' and other features.
    :param keys: List of keys to be copied from the example to the consolidated blocks.
    :return: A dictionary where each key is a bounding box and the value is a dictionary containing 'texts' and 'category'.
    """
    # Initialize unique_blocks with keys from the example
    unique_blocks = {key: example[key] for key in keys}

    # Get the list of bounding boxes
    blocks = example["bboxes_block"]

    # Initialize the 'texts' list for each unique bounding box
    for block in set(map(tuple, blocks)):
        unique_blocks[block] = {"texts": []}

    # Iterate over categories, texts and blocks in parallel
    for category, text, block in zip(example["categories"], example["texts"], blocks):
        # Convert block to a tuple so it can be used as a dictionary key
        block = tuple(block)

        # If the block doesn't have a 'category' yet, set it
        if "category" not in unique_blocks[block]:
            unique_blocks[block]["category"] = category

        # Append the text to the 'texts' list of the block
        unique_blocks[block]["texts"].append(text)

    return unique_blocks


def filter_dataset(example):
    labels = ["scientific_articles"]
    if (
        example["doc_category"] in labels
        and example["page_hash"]
        != "b2f15dd6946e4465db44572fbc734724a7db04e1c6b79f8ff6eb931a833e829c"
    ):
        return {"keep": True}
    else:
        return {"keep": False}


def load_and_filter_dataset(dataset_name="agomberto/DoCLayNet-large-wt-image"):
    """
    Loads and filters the dataset.

    :param dataset_name: The name of the dataset to load.
    :return: The filtered dataset.
    """
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(filter_dataset)
    filtered_dataset = dataset.filter(lambda example: example["keep"])

    return filtered_dataset


def consolidate_set(dataset):
    """
    Consolidates the dataset.

    :param dataset: The dataset to consolidate.
    :return: The consolidated dataset.
    """
    consolidated_set = []
    for i in tqdm(range(len(dataset))):
        consolidated_set.append(consolidate_blocks(dataset[i]))
    return consolidated_set


def get_ordered_sets(dataset, keys=["original_filename", "page_no"]):
    """
    Gets ordered sets from the dataset.

    :param dataset: The dataset to get ordered sets from.
    :param keys: The keys to use in ordering the sets.
    :return: The ordered sets.
    """
    train_dataset = dataset["train"].sort(keys)
    test_dataset = dataset["test"].sort(keys)
    val_dataset = dataset["validation"].sort(keys)

    train_set = consolidate_set(train_dataset)
    eval_set = consolidate_set(val_dataset)
    test_set = consolidate_set(test_dataset)

    return train_set, eval_set, test_set


def create_bbox(doc, _doc, key, value):
    """
    Creates a bounding box from the given document, key, and value.

    :param _doc: The document to create a bounding box from.
    :param key: The key to use in creating the bounding box.
    :param value: The value to use in creating the bounding box.
    :return: The created bounding box.
    """
    return [
        doc,
        value.get("category"),
        key[0],
        key[1],
        key[2] + key[0],
        key[3] + key[1],
        " ".join(value.get("texts")),
        len(value.get("texts")),
        0,
        _doc.get("page_no") / _doc.get("num_pages"),
        1025,
        1025,
    ]


def create_subset(set, doc):
    """
    Creates a subset from the given set.

    :param set: The set to create a subset from.
    :return: The created subset.
    """
    bboxes = []
    for _doc in set:
        for key, value in _doc.items():
            if not isinstance(key, str):
                bbox = create_bbox(doc, _doc, key, value)
                bboxes.append(bbox)
    subset = pd.DataFrame(
        bboxes,
        columns=[
            "document",
            "label",
            "x1",
            "y1",
            "x2",
            "y2",
            "text",
            "#lines",
            "#block",
            "#page",
            "width",
            "height",
        ],
    )
    return preprocess_pdf_dataset(subset)


def reorganize_data(text):
    """
    Reorganize labels bibliography

    :param text: The text to reorganize.
    :return: The reorganized label.
    """
    patterns = [
        (r"^\[(?:S|)\d+\]", 3),
        (r"^\d+(?:|\.|\)) [A-Z][\.,]", 3),
        (r"\d+[-–:\.,\s]\s*\d+(?:|\.)$", 3),
        (r"^• ", 9),
        (r"^\w+(?:|,) [A-Z][\.,]", 3),
        (r"^\d+(?:|\.|,) \w+(?:|,) [A-Z][\.,]", 3),
        (r"^[\*†∗]", 1),
        (r"\$\^\{\d+\}\$[A-Z]\.", 3),
        (r"(?:\d+:\d+|\d+ pp|\(\d+\))\.$", 3),
        (
            r"(?:arXiv:|dpo:|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",
            3,
        ),
    ]

    for pattern, value in patterns:
        if len(re.findall(pattern, text)) > 0:
            return value

    return 9


def process_dataset(dataset):
    """
    Process the dataset

    :param dataset: The dataset to process.
    :return: The processed dataset.
    """
    consolidated_dataset = consolidate_dataset(dataset)
    consolidated_dataset.loc[
        consolidated_dataset.loc[:, "label"] == 3, "label"
    ] = consolidated_dataset.loc[
        consolidated_dataset.loc[:, "label"] == 3, "text"
    ].apply(
        reorganize_data
    )
    return consolidated_dataset


def prepare_data(dataset, features):
    """
    Generate the data to be used for training.

    :param dataset: The dataset to generate the data from.
    :param features: The features to use in generating the data.
    :return: The generated data.
    """

    X = dataset.replace([np.inf, -np.inf], np.nan).dropna().loc[:, features].values
    y = (
        dataset.replace([np.inf, -np.inf], np.nan).dropna().loc[:, "label"].values
        if "label" in dataset.columns
        else None
    )

    return X, y


def consolidate_dataset(dataset):
    """
    Consolidates the dataset.

    :param dataset: The dataset to consolidate.
    :return: The consolidated dataset.
    """
    dataset_df = pd.DataFrame()
    documents = np.unique([x["original_filename"] for x in dataset])

    for doc in tqdm(documents):
        set = [x for x in dataset if x["original_filename"] == doc]
        subset = create_subset(set, doc)
        dataset_df = pd.concat([dataset_df, subset])

    return dataset_df


def normalize_position_features(df):
    """
    Normalizes the features of the dataframe.

    :param df: The dataframe to normalize.
    :return: The normalized dataframe.
    """
    df.loc[:, "x_1_normalized"] = df.x1 / df.width
    df.loc[:, "y_1_normalized"] = df.y1 / df.height
    df.loc[:, "x_2_normalized"] = df.x2 / df.width
    df.loc[:, "y_2_normalized"] = df.y2 / df.height
    df.loc[:, "height_normalized"] = (df.y2 - df.y1) / df.height
    df.loc[:, "width_normalized"] = (df.x2 - df.x1) / df.width

    df.loc[:, "area_normalized"] = df.height_normalized * df.width_normalized
    # count how many times the same bbox appears in different pages
    df.loc[:, "bboxes_rounded"] = [
        tuple(x[0]) for x in zip(df.loc[:, ["x1", "y1", "x2", "y2"]].values.round(2))
    ]
    df.loc[:, "bboxes_rounded_count_normalized"] = df.bboxes_rounded.map(
        df.bboxes_rounded.value_counts()
    ) / len(df)
    df.loc[
        :, "distance_to_previous_normalized"
    ] = df.y_1_normalized - df.y_1_normalized.shift(1).fillna(0)
    df.loc[:, "distance_to_next_normalized"] = (
        df.y_1_normalized.shift(-1).fillna(0) - df.y_1_normalized
    )

    return df


def preprocess_text(df):
    """
    Preprocesses the text of the dataframe.

    :param df: The dataframe to preprocess.
    :return: The preprocessed dataframe.
    """
    regex = r"^\$\^\{(.*?)\}\$"
    df.loc[:, "text"] = df.text.str.replace(regex, r"\1", regex=True, case=False)
    for char_open, char_close in zip(["(", "[", "〈", "{"], [")", "]", "〉", "}"]):
        regex_1 = r"\s+\{}\s*".format(char_open)
        regex_2 = r"\s+\{}".format(char_close)
        df.loc[:, "text"] = df.text.str.replace(
            regex_1, char_open, regex=True, case=False
        )
        df.loc[:, "text"] = df.text.str.replace(
            regex_2, char_close, regex=True, case=False
        )

    df.loc[:, "text"] = df.text.str.replace(r"\s+,", ",", regex=True, case=False)

    return df


def extract_caption_features(df):
    """
    Extracts caption features from the dataframe.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """

    regex = r"^(?:fig\.|figure|table|analysis|panel|measure|sfig|sfig\.|algorithm)\s+(?:III|II|IV|VIII|VII|VI|IX|X|I|V|[A-H]|\d+)(?:\.\d+|)\b(?:\s*[\.:-—]|(?:\s*[\(“][^)]*[\)”])?(?=\s+\w+[^s\s]\b))"
    df.loc[:, "starts_with_table_fig_normalized"] = df.text.str.contains(
        regex, case=False
    ).astype(float)

    return df


def extract_number_features(df, text_lengths, word_counts):
    """
    Extracts number features from the dataframe.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """

    regex = r"^\d+ [A-Z]\w+"
    df.loc[
        :, "starts_with_number_and_capital_letter_normalized"
    ] = df.text.str.contains(regex, case=False).astype(float)
    regex = r"^\d+[a-zA-Z]"
    df.loc[:, "starts_with_number_and_letter_stuck_normalized"] = df.text.str.contains(
        regex, case=False
    ).astype(float)
    regex = r"^\[\d+\]|^\d+ [A-Z]"
    df.loc[:, "starts_with_number_pattern_normalized"] = df.text.str.contains(
        regex
    ).astype(float)
    # number of numbers.
    regex = r"\d+"
    counts = df.text.str.count(regex).astype(float)
    df.loc[:, "count_number_normalized"] = np.divide(
        counts,
        word_counts,
        out=np.zeros_like(word_counts, dtype=float),
        where=word_counts != 0,
    )

    # pattern of table numbers:
    regex_numbers = r"\d{1,4}[\.,]\d{1,3}"
    number_count = df.text.str.count(regex_numbers)
    df.loc[:, "number_table_count_normalized"] = np.divide(
        number_count,
        text_lengths,
        out=np.zeros_like(word_counts, dtype=float),
        where=text_lengths != 0,
    )
    regex_numbers = r"(((?:-|)\d+(?:|[\.,]\d+)\s){2,})"
    sequences_counts = df.text.apply(lambda x: len(re.findall(regex_numbers, x)))
    df["sequence_numbers_normalized"] = np.divide(
        sequences_counts,
        text_lengths,
        out=np.zeros_like(word_counts, dtype=float),
        where=text_lengths != 0,
    )

    return df


def extract_text_features(df, text_lengths, word_counts):
    """
    Extracts text features from the dataframe.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """
    regex = r"^(?:Appendix|Introduction|Abstract|Acknowled|References|Results|Contents|Conclusion|Final remarks|Examples|Proofs)"
    df.loc[:, "starts_with_title_normalized"] = df.text.str.contains(
        regex, regex=True, case=False
    ).astype(float)
    regex = r"^[A-H1-9]+(?:0|)(?:\.|)\s+"
    df.loc[:, "starts_with_title_one_pattern_normalized"] = df.text.str.contains(
        regex, regex=True
    ).astype(float)
    regex = r"^[A-H1-9]+(?:0|)\.[A-H1-9](?:0|)[\s\.]+"
    df.loc[:, "starts_with_title_two_pattern_normalized"] = df.text.str.contains(
        regex, regex=True
    ).astype(float)
    regex = r"^(?:[IVX]+(?:\.|\s+))+"
    df.loc[:, "starts_with_roman_pattern_normalized"] = df.text.str.contains(
        regex, regex=True
    ).astype(float)
    # patterns for special characters for footnotes
    characters = ["∗", "*", "$", "‡", "†", "§", "¶"]
    regex = "^[" + "".join(characters) + "]"
    df.loc[:, "starts_with_special_char_normalized"] = df.text.str.contains(
        regex, case=False
    ).astype(float)
    # Pattern start by [A-Z]\w+, [A-Z].Field, R.
    regex = r"^[A-Z]\w+(?:,|) [A-Z](?:,|\.)"
    df.loc[:, "starts_with_name_normalized"] = df.text.str.contains(
        regex, regex=True
    ).astype(float)
    regex = r"[A-Z](?:\w+|\.)(?:\s[A-Z]\.|) [A-Z]\w+"
    names_count = df.text.str.count(regex)
    df.loc[:, "starts_with_number_name_normalized"] = np.divide(
        names_count,
        word_counts,
        out=np.zeros_like(word_counts, dtype=float),
        where=word_counts != 0,
    )
    # pattern for spaces
    regex_spaces = r"\s"
    space_count = df.text.str.count(regex_spaces)
    df.loc[:, "space_count_normalized"] = np.divide(
        space_count,
        text_lengths,
        out=np.zeros_like(word_counts, dtype=float),
        where=text_lengths != 0,
    )
    return df


def extract_same_text_features(df):
    """
    Extract the number of time the same text appears in the dataset.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """

    df.loc[:, "text_pp"] = df.text.str.replace(
        r"^VoL.\d+\s*No.\d+\s*", "", regex=True, case=False
    ).copy()
    df.loc[:, "text_pp"] = df.text_pp.str.replace(
        r"^(?:\d+|III|II|IV|VIII|VII|VI|IX|V|I|X)(?:\.|)", "", regex=True, case=False
    ).copy()
    df.loc[:, "text_pp"] = (
        df.text_pp.str.replace(r"\n", " ", regex=True).map(str.strip).str.lower()
    )
    df.loc[:, "text_pp"] = (
        df.text_pp.str.replace(r"\d+", " ", regex=True).map(str.strip).str.lower()
    )
    # same texts existing in different pages - special for num pages footer/header for instance
    word_counts_pp = df.text_pp.str.replace(
        r"(?:appendix|table|fig|figure|panel|measure|algorithm)",
        "",
        regex=True,
        case=False,
    ).map(
        df.text_pp.str.replace(
            r"(?:appendix|table|fig|figure|panel|measure|algorithm)",
            "",
            regex=True,
            case=False,
        ).value_counts()
    )
    total_page = df["#page"].unique().max() + 1
    df.loc[:, "text_wt_num_count"] = word_counts_pp.astype(float) / total_page
    df.loc[
        df.loc[:, "text_wt_num_count"] > 1, "text_wt_num_count"
    ] = 0.0  # to avoid empty values

    return df


def extract_word_features(df):
    """
    Extracts word features from the dataframe.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """
    df.loc[:, "#words"] = df["text"].str.split().apply(len)

    df.loc[:, "#words_normalized"] = (
        df.loc[:, "#words"].astype(float) / df.loc[:, "#words"].max()
    )

    df.loc[:, "#page_normalized"] = np.sign(df["#page"])

    df.loc[:, "#capitalized_words_by_block_normalized"] = np.divide(
        df.text.str.count(r"\b[A-Z][A-Za-z]+\b").astype(float),
        df.loc[:, "#words"],
        out=np.zeros_like(df.loc[:, "#words"], dtype=float),
        where=df.loc[:, "#words"] != 0,
    )
    return df


def extract_word_size_features(df):
    """
    Extracts word size features from the dataframe.

    :param df: The dataframe to extract features from.
    :return: The dataframe with the extracted features.
    """
    df.loc[:, "bbox_line_height_normalized"] = np.divide(
        df["height_normalized"],
        df["#lines"],
        out=np.zeros_like(df["#lines"], dtype=float),
        where=df["#lines"] != 0,
    )
    df.loc[
        (df.height_normalized > (df.width_normalized * 2)).values
        * (df.loc[:, "#lines"] < 10).values,
        "bbox_line_height_normalized",
    ] = np.divide(
        df["width_normalized"],
        df["#lines"],
        out=np.zeros_like(df["#lines"], dtype=float),
        where=df["#lines"] != 0,
    )

    return df


def preprocess_pdf_dataset(df):
    """
    Preprocesses the PDF dataset.

    :param df: The dataframe to preprocess.
    :return: The preprocessed dataframe.
    """
    text_lengths = df.text.map(len)
    word_counts = df.text.str.split().apply(len)
    df = df.reset_index(drop=True)
    df = preprocess_text(df)
    df = normalize_position_features(df)
    df = extract_caption_features(df)
    df = extract_number_features(df, text_lengths, word_counts)
    df = extract_text_features(df, text_lengths, word_counts)
    df = extract_same_text_features(df)
    df = extract_word_features(df)
    df = extract_word_size_features(df)

    scaler = MinMaxScaler()

    features = [col for col in df.columns if "normalized" in col]

    df.loc[:, features] = scaler.fit_transform(df.loc[:, features].values.astype(float))

    return df


def downsample_training_set(
    training_set, features, th=0.98, labels=[2, 3, 9], verbose=True
):
    """
    Downsamples the training set regarding the labels depending on the cosine similarity of different observations

    :param training_set: The training set to downsample.
    :param features: The features to use in downsampling the training set.
    :param th: The threshold to use in downsampling the training set.
    :param labels: The labels to use in downsampling the training set.
    :param verbose: Whether to print the label value counts.
    """
    _training_set = pd.DataFrame()
    if verbose:
        logging.info(training_set.label.value_counts())

    for document in training_set.loc[:, "document"].unique():
        other = (
            training_set.loc[
                (training_set.loc[:, "document"] == document)
                * (~training_set.loc[:, "label"].isin(labels)),
                :,
            ]
            .copy()
            .reset_index(drop=True)
        )
        subsets = [other]
        for i in labels:
            subset = (
                training_set.loc[
                    (training_set.loc[:, "document"] == document)
                    * (training_set.loc[:, "label"] == i),
                    :,
                ]
                .copy()
                .reset_index(drop=True)
            )
            if len(subset) > 0:
                X = subset.loc[:, features].values
                sim = cosine_similarity(X)
                ind_to_quit = np.unique(np.where(np.triu(sim, 1) > th)[1])
                subset = subset.loc[~subset.index.isin(ind_to_quit), :]
                subsets.append(subset)
        final_set = pd.concat(subsets)
        _training_set = pd.concat([_training_set, final_set]).reset_index(drop=True)

    if verbose:
        logging.info(_training_set.label.value_counts())

    return _training_set


def plot_confusion_matrix(y_true, y_pred, labels_str, id2label):
    """
    Plots a confusion matrix and a bar chart for the given true and predicted labels.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param labels_str: The string labels.
    :param id2label: The mapping from id to label.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    sns.heatmap(
        cm,
        ax=ax[0],
        annot=True,
        fmt=".2f",
        xticklabels=labels_str,
        yticklabels=labels_str,
    )
    ax[0].set_xlabel("Prédit")
    ax[0].set_ylabel("Réel")

    categories, counts = np.unique(y_true, return_counts=True)

    ax[1].bar([id2label[category] for category in categories], counts)
    ax[1].set_xlabel("Catégorie")
    ax[1].set_ylabel("Nombre")
    ax[1].set_title("Nombre d'occurrences par catégorie")
    ax[1].set_xticks(np.arange(len(categories)))
    ax[1].set_xticklabels([id2label[category] for category in categories], rotation=45)

    plt.show()


def plot_report_bar_chart(y_true, y_pred, labels_str):
    """
    Plots a bar chart for the precision, recall, and F1 score of the given true and predicted labels.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param labels_str: The string labels.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = list(report.keys())[:-3]  # Exclure 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[label]["precision"] for label in labels]
    recall = [report[label]["recall"] for label in labels]
    f1_score = [report[label]["f1-score"] for label in labels]

    macro_avg = report["macro avg"]
    micro_avg = report["weighted avg"]

    metrics = [
        ("Precision", precision, "precision"),
        ("Recall", recall, "recall"),
        ("F1 Score", f1_score, "f1-score"),
    ]
    colors = ["r", "g"]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    for i, (title, metric, key) in enumerate(metrics):
        axs[i].bar(labels_str, metric, label=title)
        axs[i].set_xticks(np.arange(len(labels)))
        axs[i].set_xticklabels(labels_str, rotation=45)
        for j, (avg, leg) in enumerate([(macro_avg, "macro"), (micro_avg, "micro")]):
            axs[i].axhline(
                avg[key],
                color=colors[j],
                linestyle="--",
                label=f"{leg} avg - {avg[key]:0.2f}",
            )
        axs[i].legend()
        axs[i].set_title(title)

    plt.tight_layout()
    plt.show()


def extract_bboxes(doc):
    """
    Extracts bounding boxes from the given document.

    :param doc: The document to extract bounding boxes from.
    :return: The extracted bounding boxes.
    """
    bboxes = []

    for i, page in enumerate(doc):
        bbox = [
            list(x)[:5] + [x[4].count("\n")] + list(x[5:6]) + [i]
            for x in page.get_text("blocks")
        ]
        bbox = [tuple(x + list(page.rect)[2:]) for x in bbox]
        bboxes += bbox

    return bboxes


def prepare_inference_data(bboxes, name=None):
    """
    Prepares data for inference.

    :param bboxes: The bounding boxes to prepare data from.
    :name: The name of the document.
    :return: The prepared data.
    """

    df = pd.DataFrame(
        bboxes,
        columns=[
            "x1",
            "y1",
            "x2",
            "y2",
            "text",
            "#lines",
            "#block",
            "#page",
            "width",
            "height",
        ],
    )

    for col in ["x1", "y1", "x2", "y2", "width", "height"]:
        df.loc[:, f"true_{col}"] = df.loc[:, col]

    if name:
        df.loc[:, "index_row"] = df.loc[:, ["#page", "y1", "x1", "y2", "x2"]].apply(
            lambda x: f"{name[:50]}_{x[0]}_{x[1]}_{x[2]}_{x[3]}_{x[4]}", axis=1
        )

    df.loc[:, "x1"] = df.loc[:, "x1"] * 1025 / df.loc[:, "width"]
    df.loc[:, "x2"] = df.loc[:, "x2"] * 1025 / df.loc[:, "width"]
    df.loc[:, "y1"] = df.loc[:, "y1"] * 1025 / df.loc[:, "height"]
    df.loc[:, "y2"] = df.loc[:, "y2"] * 1025 / df.loc[:, "height"]

    df.loc[:, "text"] = df.text.map(str.strip)
    df.loc[:, "text"] = df.text.str.replace("\n", " ").map(str.strip)
    df.loc[:, "width"] = 1025
    df.loc[:, "height"] = 1025
    df = (
        df.loc[df.text.replace("\d+", "", regex=True).map(str.strip).str.len() > 0, :]
        .reset_index(drop=True)
        .copy()
    )
    df = preprocess_pdf_dataset(df)

    features = [col for col in df.columns if "normalized" in col]

    scaler = MinMaxScaler()

    X = df.replace([np.inf, -np.inf], np.nan).fillna(0).loc[:, features].values
    X = scaler.fit_transform(X)

    return X, df


def prepare_tfidf_data(training_set, validation_set, features, tf_idf=None):
    """
    Prepares the data and add tf_idf features if needed.

    :param training_set: The training set to prepare the data from.
    :param validation_set: The validation set to prepare the data from.
    """
    X_train, y_train = prepare_data(training_set, features)
    X_val, y_val = prepare_data(validation_set, features)

    if tf_idf:
        tfidf = TfidfVectorizer(
            min_df=3,
            max_df=0.7,
            ngram_range=(1, 2),
            stop_words="english",
            max_features=1000,
        )

        X_text = tfidf.fit_transform(training_set["text"])
        X_val_text = tfidf.transform(validation_set["text"])

        X_train = np.concatenate([X_train, X_text.toarray()], axis=1)
        X_val = np.concatenate([X_val, X_val_text.toarray()], axis=1)
    else:
        tfidf = None

    return tfidf, X_train, X_val, y_train, y_val


def post_process_inference_data(df, y_pred):
    """
    Post-processes the inference data.

    :param df: The dataframe to post-process.
    :param y_pred: The predicted labels.
    :return: The post-processed dataframe.
    """

    df.loc[:, "label_preds"] = y_pred

    df = post_process_footnote_height(df)
    df = post_process_header_footer(df)
    df = post_process_footnotes_sequence(df)
    df = post_process_bibliography(df)
    df = post_process_header_table(df)
    df = post_process_header_bibliography(df)
    df = post_process_caption(df)
    df = post_process_text_between_caption_image(df)
    df = post_process_section_header_for_parsing(df)

    return df


def post_process_footnote_height(df, threshold=0.85):
    """
    Post-processes the footnote height in the given dataframe.

    :param df: The dataframe to post-process.
    :param threshold: The threshold for the normalized bounding box line height.
    :return: The post-processed dataframe.
    """
    height_series = []
    for height, weight in df.loc[
        df.loc[:, "label_preds"].isin([1, 2, 4, 5, 7, 9, 10]),
        ["bbox_line_height_normalized", "#lines"],
    ].values:
        height_series += [height] * int(weight)

    # footnote specific
    df.loc[
        (
            df.loc[:, "bbox_line_height_normalized"]
            < threshold * np.median(height_series)
            + df.text.str.contains("^\d+[A-Z]", regex=True)
        )
        * (df.loc[:, "label_preds"] == 9),
        "label_preds",
    ] = 1

    return df


def post_process_header_footer(df, threshold=0.2):
    """
    Post-processes the header and footer of the dataframe. if the bbox is fixed in the page more than threshold%, it is a header or footer

    :param df: The dataframe to post-process.
    :param threshold: The threshold to use in post-processing the dataframe.
    :return: The post-processed dataframe.
    """

    df.loc[
        (df.loc[:, "text_wt_num_count"] > threshold)
        * (df.loc[:, "label_preds"].isin([1, 3, 7, 9])),
        "label_preds",
    ] = 4

    df.loc[
        (df.text.str.split().map(len) > 11) * (df.label_preds == 7), "label_preds"
    ] = 9

    return df


def post_process_footnotes_sequence(df):
    """
    Post-processes the footnotes sequence of the dataframe. if  text is considered as footnote but the next block is not, it is not a footnote

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """
    # footnote specific if one footnote between texts should be text
    footnotes = df.loc[df.loc[:, "label_preds"] == 1, "text"].values
    for footnote in footnotes:
        page, block = df.loc[df.text == footnote, ["#page", "#block"]].values[0]
        ind = df.loc[df.text == footnote, :].index[0]
        subset = df.loc[
            (df.loc[:, "#page"] == page), ["text", "#page", "#block", "label_preds"]
        ].copy()
        if 9 in subset.loc[ind + 1 :, "label_preds"].values:
            df.loc[ind, "label_preds"] = 9

    return df


def post_process_bibliography(df):
    """
    Post-processes the bibliography of the dataframe. if  text is considered as a text but is between bibliography lines, it is a bibliography

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """
    # text specific if one text between bibliography should be bibliography
    texts = df.loc[df.loc[:, "label_preds"] == 9, "text"].values
    for text in texts:
        page, block = df.loc[df.text == text, ["#page", "#block"]].values[0]
        ind = df.loc[df.text == text, :].index[0]
        subset = df.loc[
            (df.loc[:, "#page"] == page), ["text", "#page", "#block", "label_preds"]
        ].copy()
        if (
            3 in subset.loc[ind + 1 :, "label_preds"].values
            and 3 in subset.loc[: ind - 1, "label_preds"].values
        ):
            df.loc[ind, "label_preds"] = 3

    return df


def post_process_header_table(df):
    """
    Postprocess the header table of the dataframe that should be a table if have specific words

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """

    # header that are part of table
    headers = df.loc[df.loc[:, "label_preds"].isin([4, 5]), "text"].values
    for header in headers:
        words = [
            "statistics",
            "mean",
            "score",
            "value",
            "hypoth",
            "treatment",
            "category",
            "observation",
            "cluster",
            "panel",
            "control",
            "data collection",
            "total",
        ]
        matches = re.findall(r"(?:{})".format("|".join(words)), header, re.IGNORECASE)
        if len(matches) > 0:
            df.loc[df.text == header, "label_preds"] = 8

    return df


def post_process_header_bibliography(df):
    """
    Post-processes the references bibliography of the dataframe. between the reference section and next section header
    everything should be a bibliography

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """

    # bibliography specific
    ind = df.loc[
        (df.text.str.contains("References", case=False)) * (df.label_preds == 7), :
    ]
    if len(ind) > 0:
        ind = ind.index[0]
        ind_next_section = df.loc[ind + 1 :, :].loc[
            df.loc[ind + 1 :, :].loc[:, "label_preds"].isin([0, 2, 6, 8, 7]), :
        ]
        if len(ind_next_section) > 0:
            ind_next_section = ind_next_section.index[0]
        else:
            ind_next_section = df.shape[0]
        df.loc[ind + 1 : ind_next_section - 1, "label_preds"] = 3

    return df


def post_process_caption(df):
    """
    Post process potential captions. If there is table at some points in a header or text (that begins) it's most likely a caption

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """
    df.loc[
        (df.text.str.contains(r"^(?:table|figure)", regex=True, case=False))
        * (df.label_preds == 7),
        "label_preds",
    ] = 0
    df.loc[
        (df.text.str.contains(r"^(?:table|figure)", regex=True, case=False))
        * (df.label_preds == 1),
        "label_preds",
    ] = 0
    df.loc[
        (
            df.text.str.contains(
                r"^(?:fig\.|table)\s+(?:III|II|IV|VIII|VII|VI|IX|X|I|V|[A-H]|\d+)(?:\.\d+|)\b[\.:-—]",
                regex=True,
                case=False,
            )
        )
        * (df.label_preds == 9),
        "label_preds",
    ] = 0

    return df


def check_order(s):
    """
    Checks the order of certain characters in the given string.

    :param s: The string to check.
    :return: True if the characters "6", "8", and "2" appear after "0", False otherwise.
    """
    caption_first = True
    if "6" in s:
        caption_first *= s.find("6") > s.find("0")
    if "8" in s:
        caption_first *= s.find("8") > s.find("0")
    if "2" in s:
        caption_first *= s.find("2") > s.find("0")
    return bool(caption_first)


def post_process_text_between_caption_image(df):
    """
    Post process the text between caption and image. If the text is between caption and image, it's most likely a table

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """
    captions = df.loc[
        (df.text.str.contains("table", case=False)) * (df.label_preds == 0), "text"
    ].values
    for caption in captions:
        page, block = df.loc[df.text == caption, ["#page", "#block"]].values[0]
        ind = df.loc[df.text == caption, :].index[0]
        subset = df.loc[
            (df.loc[:, "#page"] == page),
            [
                "text",
                "x1",
                "x2",
                "y1",
                "y2",
                "#page",
                "count_number_normalized",
                "#block",
                "label_preds",
            ],
        ].copy()
        horizontal = (
            True
            if df.loc[ind, "height_normalized"] > (df.loc[ind, "width_normalized"] * 2)
            else False
        )
        subset.sort_values(
            by=["x2", "y1"], inplace=True
        ) if horizontal else subset.sort_values(by=["y2", "x2"], inplace=True)
        caption_first = check_order(
            "".join(subset.loc[:, "label_preds"].astype(str).values.tolist())
        )
        if not caption_first or ind == subset.index[-1]:
            subset = subset[::-1]
        list_ind = subset.loc[subset.label_preds.isin([2, 6, 8]), :].index
        if len(list_ind) > 0:
            _ind = list_ind[-1]
        else:
            continue
        final_ind = subset.loc[ind:_ind, :].index
        final_ind = [
            x
            for x in final_ind[1:]
            if len(re.findall("^(?:Note|Analysis Note|\*)", subset.loc[x, "text"])) == 0
        ]
        df.loc[final_ind, "label_preds"] = 8
        if len(final_ind) > 0:
            ind = final_ind[0] - 1
            if ind in df.index and re.findall(
                "^(?:Note|Analysis Note|\*)", df.loc[ind, "text"]
            ):
                df.loc[ind, "label_preds"] = 1
            ind = final_ind[-1] + 1
            if ind in df.index and re.findall(
                "^(?:Note|Analysis Note|\*)", df.loc[ind, "text"]
            ):
                df.loc[ind, "label_preds"] = 1

    return df


def save_plot_with_bbox(pdf, doc, df, PATH_DATA):
    """
    Saves the plot with bounding boxes.

    :param pdf: The PDF to save the plot with bounding boxes for.
    :param doc: The document to save the plot with bounding boxes for.
    :param df: The dataframe to save the plot with bounding boxes for.
    :param PATH_DATA: The path to save the plot with bounding boxes for.
    """

    label2color = {
        "Caption": "brown",
        "Footnote": "orange",
        "Formula": "gray",
        "Bibliography": "yellow",
        "Page-footer": "red",
        "Page-header": "red",
        "Picture": "violet",
        "Section-header": "orange",
        "Table": "green",
        "Text": "blue",
        "Title": "pink",
    }

    id2label = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "Bibliography",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title",
    }

    for i, page in enumerate(doc):
        subdf = df.loc[df.loc[:, "#page"] == i, :].copy()
        for index, row in subdf.iterrows():
            bbox = (row["true_x1"], row["true_y1"], row["true_x2"], row["true_y2"])
            predicted_label = id2label[row["label_preds"]]
            page.add_freetext_annot(
                fitz.Rect(bbox),
                predicted_label,
                border_color=getColor(label2color[predicted_label]),
                fontsize=5,
                align=1,
            )
    doc.save(
        PATH_DATA + f"/{pdf}_rule_based_with_bbox.pdf",
        garbage=4,
        deflate=True,
        clean=True,
    )


def post_process_section_header_for_parsing(df):
    """
    Post-processes the section headers in the given dataframe for parsing.

    :param df: The dataframe to post-process.
    :return: The post-processed dataframe.
    """

    df.loc[:, "hierarchy"] = None
    df.loc[:, "integer_true"] = False

    reference = df.text.str.contains(
        "^(?:references|bibliography)", regex=True, case=False
    )
    if reference.sum() > 0:
        reference_index = df.loc[reference, :].index[0]
    else:
        reference_index = df.shape[0]

    subset = df.loc[:reference_index, :]
    subset.loc[subset.loc[:, "label_preds"] == 7, "integer_true"] = subset.loc[
        subset.loc[:, "label_preds"] == 7, "text"
    ].str.contains("^(?:[1-9IixXvV]+|[A-H])[-,\s\.]", regex=True)

    subset.loc[subset.loc[:, "label_preds"] == 7, :] = create_hierarchy(
        subset.loc[subset.loc[:, "label_preds"] == 7, :]
    )

    indexes_true = subset.loc[
        (subset.loc[:, "label_preds"] == 7) * (subset.loc[:, "integer_true"]), "text"
    ].index
    indexes_false = [
        x
        for x in subset.loc[(subset.loc[:, "label_preds"] == 7), :].index
        if x not in indexes_true and x < indexes_true.max()
    ]
    subset.loc[indexes_false, "label_preds"] = 9

    return df


def create_hierarchy(df):
    """
    Creates a hierarchy for the section headers in the given dataframe.

    :param df: The dataframe to create a hierarchy for.
    :return: The dataframe with the created hierarchy.
    """

    if df.integer_true.sum() > 0:
        for i, row in df.iterrows():
            # Extraire le préfixe numérique
            prefix = row["text"].split(" ")[0]

            # Déterminer le niveau de la hiérarchie en fonction du préfixe
            if "." not in prefix:
                level = 1
            else:
                level = prefix.count(".")

            # Mettre à jour la colonne de la hiérarchie
            df.at[i, "hierarchy"] = level
    else:  # decide mode of sizes and
        optimal_components = find_optimal_components(
            df.loc[:, "bbox_line_height_normalized"], 5
        )
        _, _, assignments = find_distributions_and_assign(
            df.loc[:, "bbox_line_height_normalized"], optimal_components
        )
        df.loc[:, "hierarchy"] = assignments

    return df


def find_optimal_components(data, max_components):
    """
    Creates a hierarchy for the section headers in the given dataframe.

    :param df: The dataframe to create a hierarchy for.
    :return: The dataframe with the created hierarchy.
    """
    # Reshape data for the model
    data = np.array(data).reshape(-1, 1)
    if len(data) < 2:
        return 1
    max_components = min(max_components, len(data))

    # Calculate BIC for different number of components
    bics = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n)
        gmm.fit(data)
        bics.append(gmm.bic(data))

    # Find the number of components that gives the lowest BIC
    optimal_components = np.argmin(bics) + 1

    return optimal_components


def find_distributions_and_assign(data, n_components):
    """
    Assigns data to Gaussian distributions and returns the means, standard deviations, and assignments.

    :param data: The data to assign to distributions.
    :param n_components: The number of Gaussian distributions.
    :return: The means, standard deviations, and assignments.
    """
    # Reshape data for the model
    data = np.array(data).reshape(-1, 1)
    if len(data) < 2:
        return np.nan, np.nan, np.ones(len(data))

    # Create and fit the model
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)

    # Get the means, standard deviations, and assignments
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    assignments = gmm.predict(data)

    # Sort by means
    sorted_indices = np.argsort(means)[::-1]  # Sort in descending order
    means = means[sorted_indices]
    stds = stds[sorted_indices]
    assignments = (
        np.array([np.where(sorted_indices == a)[0][0] for a in assignments]) + 1
    )

    return means, stds, assignments


def extract_text_elements(subsubset):
    """
    Extracts text, footnotes, bibliography, and tables from the given subset of a dataframe.

    :param subset: The subset of the dataframe to extract elements from.
    :return: A dictionary containing the extracted text, footnotes, bibliography, and tables.
    """
    text = subsubset.loc[subsubset.loc[:, "label_preds"] == 9, "text"].values.tolist()
    text = " \n\n ".join(text) if len(text) > 0 else ""
    footnotes = subsubset.loc[
        subsubset.loc[:, "label_preds"] == 1, "text"
    ].values.tolist()
    footnotes = " \n\n ".join(footnotes) if len(footnotes) > 0 else ""
    biblio = subsubset.loc[subsubset.loc[:, "label_preds"] == 3, "text"].values.tolist()
    biblio = " \n\n ".join(biblio) if len(biblio) > 0 else ""
    tables = subsubset.loc[
        subsubset.loc[:, "label_preds"].isin([0, 8]), "text"
    ].values.tolist()
    tables = " \n\n ".join(tables) if len(tables) > 0 else ""

    my_dict = {
        "text": text,
        "footnotes": footnotes,
        "bibliography": biblio,
        "tables": tables,
    }

    return my_dict


def get_parsed_file(df):
    """
    Parses a document represented as a dataframe into a dictionary structure.

    :param df: The dataframe representing the document.
    :return: A dictionary representing the parsed document.
    """

    final_parsing = {}
    if df.loc[df.loc[:, "label_preds"] == 10, "text"].shape[0] > 0:
        final_parsing["title"] = " ".join(
            df.loc[df.loc[:, "label_preds"] == 10, "text"].values
        )

    subset = df.loc[~df.loc[:, "label_preds"].isin([4, 5, 10]), :].copy()
    subset = subset.loc[
        subset.text.replace("\d+", "", regex=True).map(str.strip).str.len() > 0, :
    ].copy()

    reference = df.text.str.contains(
        "^(?:references|bibliography)", regex=True, case=False
    )
    if reference.sum() > 0:
        reference_index = df.loc[reference, :].index[0]
        section_header_indexes = list(
            subset.loc[:reference_index, :]
            .loc[subset.loc[:reference_index, "label_preds"] == 7, :]
            .index
        )
        other_section_header_indexes = list(
            subset.loc[reference_index:, :]
            .loc[subset.loc[reference_index:, "label_preds"] == 7, :]
            .index
        )
    else:
        section_header_indexes = list(
            subset.loc[subset.loc[:, "label_preds"] == 7, :].index
        )
        other_section_header_indexes = []
    levels = {}
    for i, row_i in enumerate(section_header_indexes):
        if i == 0:
            subsubset = subset.loc[:row_i, :].copy()
            if len(subsubset) == 0:
                continue
            key = "executive_summary"
            hierarchy = 1
            my_dict = extract_text_elements(subsubset)
            final_parsing[key] = my_dict

        if i < len(section_header_indexes) - 1:
            row_i_1 = section_header_indexes[i + 1]
            subsubset = subset.loc[row_i + 1 : row_i_1 - 1, :].copy()
        else:
            subsubset = subset.loc[row_i:, :].copy()

        key = subset.loc[row_i, "text"]
        hierarchy = subset.loc[row_i, "hierarchy"]

        my_dict = extract_text_elements(subsubset)

        if hierarchy == 1 or i == 0:
            final_parsing[key] = my_dict
            levels[1] = key
        else:
            levels[hierarchy] = key
            temp = final_parsing
            for i in range(1, hierarchy):
                try:
                    temp = temp[levels[i]]
                except KeyError:
                    levels[i] = levels[i - 1]
                    temp[levels[i]] = {}
                    temp = temp[levels[i]]
            temp[key] = my_dict

    for i, row_i in enumerate(other_section_header_indexes):
        final_parsing["after_references"] = {}
        if i < len(other_section_header_indexes) - 1:
            row_i_1 = other_section_header_indexes[i + 1]
            subsubset = subset.loc[row_i + 1 : row_i_1 - 1, :].copy()
        else:
            subsubset = subset.loc[row_i:, :].copy()

        key = subset.loc[row_i, "text"]
        hierarchy = 1
        my_dict = extract_text_elements(subsubset)
        final_parsing["after_references"][key] = my_dict

    return final_parsing


def load_pdf(PATH_DATA: str, file: str):
    """
    Converts a PDF file into a list of images.

    :param PATH_DATA: Path to the directory containing the PDF file.
    :param file: Name of the PDF file without the extension.
    :return: document object.
    """

    logging.info(f"Loading PDF from {file}...")

    # Open the PDF file
    doc = fitz.open(join(PATH_DATA, f"{file}.pdf"))

    return doc


def process_document_for_captions(doc, model_path):
    """
    Process the given document to extract bounding boxes, perform predictions, and filter the data.

    Args:
    doc (object): The document object to be processed.
    model_path (str): Path to the serialized classifier model.

    Returns:
    DataFrame: Filtered dataframe with predictions.
    """
    # Function to extract bounding boxes from the document
    bboxes = extract_bboxes(doc)

    # Prepare the data for inference
    X, cap_df = prepare_inference_data(bboxes)

    # Load the pre-trained classifier
    with open(model_path, "rb") as model_file:
        clf = pickle.load(model_file)

    # Make predictions on the data
    y_preds = clf.predict(X)

    # Post-process inference data
    cap_df = post_process_inference_data(cap_df, y_preds)

    # Filter the dataframe based on predictions
    cap_df_filtered = cap_df[cap_df["label_preds"] == 0]
    # cap_df_filtered = cap_df[(cap_df['label_preds'] == 0) | (cap_df['label_preds'] == 8)]

    return cap_df_filtered
