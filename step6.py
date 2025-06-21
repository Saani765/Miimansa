import os
import re
import json
from pprint import pprint
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

def parse_original_ann(ann_file):
    """
    Parses an .ann file from the 'original' directory.
    Extracts entity annotations (ID, label, span, and text).
    """
    annotations = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or not line.startswith('T'):
                continue
            parts = line.strip().split('\t')
            if len(parts) == 3:
                ann_id, label_span, text = parts
                # Format is "label start end"
                try:
                    label, start, end = label_span.split(' ', 2)
                    if ';' in end: # Handle discontinuous spans by taking the first part
                        end = end.split(';')[0]
                    annotations.append({
                        'id': ann_id,
                        'label': label,
                        'start': int(start),
                        'end': int(end),
                        'text': text
                    })
                except ValueError:
                    # Skip malformed lines
                    continue
    return annotations

def parse_sct_ann(ann_file):
    """
    Parses a .ann file from the 'sct' directory.
    Extracts entities and their associated SNOMED-CT codes.
    The format is: TT<ID> <SNOMED_CODE> | <SNOMED_TEXT> | <START> <END> <ORIGINAL_TEXT>
    """
    annotations = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('TT'):
                continue
            
            try:
                # Split the line into major components
                parts = line.split('\t')
                id_part = parts[0]
                info_part = parts[1]
                
                # Further split the info part "<code> | <text> | <spans>"
                info_components = info_part.split('|')
                snomed_code = info_components[0].strip()
                snomed_text = info_components[1].strip()

                # The label is not explicitly defined in the same way as original,
                # but we can infer it or just use a generic one if needed.
                # For now, we mainly need the code and text for matching.
                annotations.append({
                    'id': id_part,
                    'label': 'SCT_Entity', # Using a generic label
                    'snomed_code': snomed_code,
                    'snomed_text': snomed_text
                })
            except (IndexError, ValueError) as e:
                # print(f"Skipping malformed line in {ann_file}: {line} - Error: {e}")
                continue
    return annotations

def build_combined_data(file_list):
    """
    Builds a data structure combining information from 'original' and 'sct'
    directories for a given list of files.
    """
    combined_data = {}
    for txt_file in file_list:
        base = txt_file.replace('.txt', '')
        original_ann_file = os.path.join('cadec/original', base + '.ann')
        sct_ann_file = os.path.join('cadec/sct', base + '.ann')

        if os.path.exists(original_ann_file) and os.path.exists(sct_ann_file):
            original_annotations = parse_original_ann(original_ann_file)
            sct_annotations = parse_sct_ann(sct_ann_file)
            
            # Filter for ADR labels from original and find corresponding SCT info
            # For now, we store them separately and will match them later.
            combined_data[base] = {
                'original': original_annotations,
                'sct': sct_annotations
            }
    return combined_data

def match_with_fuzzywuzzy(adr_text, sct_annotations):
    """
    Finds the best SNOMED-CT match for an ADR text using fuzzy string matching.
    """
    best_match = None
    max_score = -1
    
    # Only consider sct annotations that have a snomed code
    sct_candidates = [sct for sct in sct_annotations if sct.get('snomed_code')]

    if not sct_candidates:
        return None, 0

    for sct_ann in sct_candidates:
        # token_set_ratio is good for matching phrases with different wording
        score = fuzz.token_set_ratio(adr_text, sct_ann['snomed_text'])
        if score > max_score:
            max_score = score
            best_match = sct_ann
            
    return best_match, max_score

def match_with_embeddings(adr_text, sct_annotations, model):
    """
    Finds the best SNOMED-CT match for an ADR text using sentence embeddings.
    """
    best_match = None
    max_score = -1
    
    sct_candidates = [sct for sct in sct_annotations if sct.get('snomed_code')]
    if not sct_candidates:
        return None, 0

    # Encode the ADR text
    adr_embedding = model.encode(adr_text, convert_to_tensor=True)
    
    # Encode all SCT texts
    sct_texts = [sct['snomed_text'] for sct in sct_candidates]
    sct_embeddings = model.encode(sct_texts, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(adr_embedding, sct_embeddings)
    
    # Find the highest score
    top_result = cosine_scores[0].argmax()
    max_score = cosine_scores[0][top_result].item()
    best_match = sct_candidates[top_result]
    
    return best_match, max_score

def main():
    # Load a pre-trained model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # Load the list of sampled files
    with open('step5_sampled_files.txt', 'r') as f:
        sampled_files = [line.strip() for line in f if line.strip()]

    # Build the main data structure
    data = build_combined_data(sampled_files)

    results = []
    
    print("\n--- Starting Annotation Matching ---")
    # Process all files, but we can break early for demonstration
    for i, (filename, content) in enumerate(data.items()):
        original_adrs = [ann for ann in content['original'] if ann['label'] == 'ADR']
        if not original_adrs:
            continue

        print(f"\n--- Processing File: {filename} ({i+1}/{len(data)}) ---")
        
        for adr_ann in original_adrs:
            # Method a) Fuzzy String Matching
            fuzzy_match, fuzzy_score = match_with_fuzzywuzzy(adr_ann['text'], content['sct'])

            # Method b) Embedding Similarity
            embedding_match, embedding_score = match_with_embeddings(adr_ann['text'], content['sct'], model)

            results.append({
                'file': filename,
                'original_text': adr_ann['text'],
                'fuzzy_match_text': fuzzy_match['snomed_text'] if fuzzy_match else 'N/A',
                'fuzzy_match_code': fuzzy_match['snomed_code'] if fuzzy_match else 'N/A',
                'fuzzy_score': fuzzy_score,
                'embedding_match_text': embedding_match['snomed_text'] if embedding_match else 'N/A',
                'embedding_match_code': embedding_match['snomed_code'] if embedding_match else 'N/A',
                'embedding_score': embedding_score,
            })
        
        # Display results for this file immediately
        for res in results:
            if res['file'] == filename:
                print(f"\nOriginal ADR: '{res['original_text']}'")
                print(f"  A) Fuzzy Match: '{res['fuzzy_match_text']}' (Code: {res['fuzzy_match_code']}) - Score: {res['fuzzy_score']:.2f}")
                print(f"  B) Embedding Match: '{res['embedding_match_text']}' (Code: {res['embedding_match_code']}) - Score: {res['embedding_score']:.2f}")
        
        # To keep the output manageable, let's just process one file for now.
        # Remove or comment out the 'break' to run on all sampled files.
        # break

    print("\n--- Comparison Complete ---")

if __name__ == '__main__':
    main() 