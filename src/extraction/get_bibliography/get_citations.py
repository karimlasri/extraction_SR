import os
import fitz 
import re
import pandas as pd
import argparse


regexes = {
    'author_regex' : '([A-Z]{1,2}[a-z]*(-[A-Z]{1,2}[a-z]*)?\.?(\s([A-Z]{1,2}[a-z]*(-[A-Z]{1,2}[a-z]*)?\.?)+)\,\s)+',
    'doi_regex' : '10.[0-9]{4}\/[0-9A-Za-z]*',
    'site_regex' : 'http',
    'year_regex' : '[^\w\d]((19|(2[0-2])[0-9]{2}))[^\w\d]'
}


def get_line_signal(line):
    signal = []
    for reg_tp, re_string in regexes.items():
        signal.append(len(re.findall(re_string, line)))
    signal[0]+=len(re.findall('(et al.|ET AL.)', line))
    return 20<=len(line)<=500, signal


def eval_ref_block_text(block_text):
    len_signal, reg_signal = get_line_signal(block_text)
    if len_signal:
        return sum(reg_signal[:-1])>=1
    else:
        return False


def get_chunks(is_ref_list):
    chunks = []
    cur_start = None
    cur_end = None
    for i, is_ref in enumerate(is_ref_list):
        if is_ref:
            if cur_start is None:
                cur_start=i
                if cur_end is None:
                    cur_end=i
            else:
                cur_end=i
        else:
            if cur_start is not None and cur_end is not None:
                chunks.append((cur_start, cur_end))
                cur_start = None
                cur_end = None
    if cur_start is not None and cur_end is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def merge_chunks(chunks, disc_threshold=2):
    merged_chunks = []
    previous_chunk = None
    for cur_chunk in chunks:
        if previous_chunk is None:
            previous_chunk = cur_chunk
        else:
            if previous_chunk[1]+disc_threshold>=cur_chunk[0]:
                previous_chunk = (previous_chunk[0], cur_chunk[1])
            else:
                merged_chunks.append(previous_chunk)
                previous_chunk = (cur_chunk[0], cur_chunk[1])
    merged_chunks.append(previous_chunk)
    return merged_chunks


def get_criterion(is_ref_list, length_threshold=4):
    chunks = get_chunks(is_ref_list)
    merged_chunks = merge_chunks(chunks, disc_threshold=2)
    if chunks:
        chunk_lengths = [c[1]+1-c[0] for c in merged_chunks]
        return max(chunk_lengths) >= length_threshold, merged_chunks
    else:
        return False, []
    
    
def slice_block(block_text, ratio_threshold=0.8):
    lines = block_text.split('\n')
    slices = []
    cur_slice = []
    for i, l in enumerate(lines):
        if len(l)<10:
            if not cur_slice:
                continue
        cur_slice.append(l)
        if i>=0:
            if len(l)<=len(lines[i-1])*ratio_threshold:
                slices.append(' '.join(cur_slice))
                cur_slice = []
    cur_slice_text = ' '.join(cur_slice)
    if len(cur_slice_text)>10:
        slices.append(cur_slice_text)
    return slices


def merge_block_texts(block_texts):
    returns = [len(re.findall('\n',t)) for t in block_texts]
    if max(returns)<3:
        return [''.join(block_texts)]
    else: 
        return block_texts


def eval_page(p):
	""" Tests whether a given page contains signal for potential citations. """
    blocks = p.get_text('blocks')
    block_texts = [b[4] for b in blocks]
    if len(blocks)>2:
        is_ref_list = []
        citations = []
        block_texts = merge_block_texts(block_texts)
        for block_text in block_texts:
            block_slices = slice_block(block_text)
            for block_text in block_slices:
                if len(block_text)>20:
                    is_ref = eval_ref_block_text(block_text)
                    is_ref_list.append(is_ref)

        length_threshold = max(min(len(is_ref_list)-1, 4), 2)
        criterion, merged_chunks = get_criterion(is_ref_list, length_threshold)
        for chunk in merged_chunks:
            start, end = chunk
            for j in range(start, end+1):
                citations.append(p.number)
        return criterion, citations
    else:
        return False, []


def get_citations(pdf_path):
	""" Detects all pages that contain signal indicating it could be a bibliography page. """
    all_citations = {}
    doc = fitz.open(pdf_path)
    ref_pages = []
    for i, p in enumerate(doc):
        is_ref_page, citations = eval_page(p)
        if is_ref_page:
            all_citations[i] = citations
            ref_pages.append(i)
    return all_citations, ref_pages


def evaluate(detected_pages, true_range):
    true_pages = list(range(true_range[0], true_range[1]+1))
    intersection = len(set(detected_pages).intersection(true_pages))
    precision = intersection/max(1, len(detected_pages))
    recall = intersection/max(1, len(true_pages))
    return precision, recall
    

def save_citations(base_name, out_folder, citations):
    citation_strings = [c[0] for c in citations]
    citation_pages = [c[1] for c in citations]
    citations_df = pd.DataFrame({'Citation':citation_strings, 'Page':citation_pages})
    citations_df.to_csv('{}{}.csv'.format(out_folder,base_name))


def clean_pages_signal(detected_pages, ground_truth=None):
    if detected_pages:
        detected_bool_list = [False for i in range(max(detected_pages)+1)]
        for p in detected_pages:
            detected_bool_list[p]=True
        chunks = get_chunks(detected_bool_list)
        merged_chunks = merge_chunks(chunks, 1)
        clean_chunks = [c for c in merged_chunks if c[1]!=c[0]]
        if ground_truth is not None:
            clean_chunks = [c for c in clean_chunks if c[0]>=ground_truth[0] and c[1]<=ground_truth[1]]
        clean_pages = []
        for c in clean_chunks:
            for i in range(c[0], c[1]+1):
                clean_pages.append(i)
        return clean_pages, clean_chunks
    else:
        return [], []


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Extract citation strings from raw pdf files."
    )

    parser.add_argument("--pdf_folder", type=str, help="Path to the input pdfs.")
    parser.add_argument("--out_folder", type=str, help="Path to output.")

    args = parser.parse_args()
    
    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)
    
    for pdf_name in os.listdir(args.pdf_folder):
        if pdf_name.endswith('.pdf'):
            
            pdf_path = args.pdf_folder + pdf_name
            
            # Get candidate citations and pages
            all_citations, detected_pages = get_citations(pdf_path)
    
            # Clean candidates
            clean_detected_pages, clean_page_chunks = clean_pages_signal(detected_pages)
            clean_citations = []
            for p in clean_detected_pages:
                clean_citations += all_citations[p]
    
            for c in clean_page_chunks:
                print('Found bibliography from p.{} to p.{}'.format(c[0], c[1]))
    
            base_name = pdf_name.split('.')[0]
            save_citations(base_name, args.out_folder, clean_citations)
    