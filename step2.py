import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 1. Load a suitable model and tokenizer from Hugging Face for token classification
# (e.g., 'dslim/bert-base-NER' or similar, can be changed later)
#
# 2. Define a function to read a forum post from 'cadec/text/<filename>.txt'
#
# 3. Use the model to label each word in BIO/IOB format
#
# 4. Convert the BIO/IOB output to the label format as in 'cadec/original/<filename>.ann'
#
# 5. Print or save the results for inspection

# TODO: Implement each step above, one by one, with clear comments and beginner-friendly code.

def read_forum_post(text_dir, filename):
    """
    Reads the contents of a forum post from the cadec/text directory.
    Args:
        text_dir (str): Path to the text directory (e.g., 'cadec/text')
        filename (str): Name of the file to read (e.g., 'ARTHROTEC.1.txt')
    Returns:
        str: The contents of the forum post as a string.
    """
    file_path = os.path.join(text_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def label_text_with_bio(text, model_name="d4data/biomedical-ner-all"):
    """
    Uses a Hugging Face NER pipeline to label each word in the text with BIO/IOB format.
    Args:
        text (str): The input forum post text.
        model_name (str): The Hugging Face model to use.
    Returns:
        list of dict: Each dict contains 'word', 'entity', 'score', 'start', 'end'.
    """
    # Load the model and tokenizer
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")
    # Run the pipeline on the text
    ner_results = ner_pipeline(text)
    return ner_results

def map_entity_label(entity_label):
    """
    Maps the model's entity label to one of the four required categories: ADR, Drug, Disease, Symptom.
    Returns None if the label should be ignored.
    """
    # Lowercase for robustness
    label = entity_label.lower()
    # Mapping based on typical biomedical NER conventions and CADEC .ann files
    if label in ["medication", "drug","therapeutic_procedure"]:
        return "Drug"
    elif label in ["disease_disorder", "disease"]:
        return "Disease"
    elif label in ["sign_symptom", "symptom"]:
        return "Symptom"
    elif label in ["adr", "adverse_event","sign_symptom", "adverse drug reaction", "adverse reaction","detailed_description"]:
        return "ADR"
    # Add more mappings if needed based on model output
    return None

def convert_ner_to_ann_format(ner_results, label_prefix="T"):
    """
    Converts NER results to the .ann label format used in 'cadec/original',
    mapping model labels to the four required categories.
    Args:
        ner_results (list of dict): Output from the NER pipeline.
        label_prefix (str): Prefix for the tag (default 'T').
    Returns:
        list of str: Each string is a line in the .ann format: Tag<TAB>Label Start End<TAB>Text
    """
    ann_lines = []
    idx = 1
    for entity in ner_results:
        mapped_label = map_entity_label(entity['entity_group'])
        if mapped_label is None:
            continue  # Skip entities not in the required categories
        tag = f"{label_prefix}{idx}"
        start = entity['start']
        end = entity['end']
        text = entity['word']
        ann_line = f"{tag}\t{mapped_label} {start} {end}\t{text}"
        ann_lines.append(ann_line)
        idx += 1
    return ann_lines

def postprocess_ner_results(ner_results, text):
    """
    Postprocesses the NER results to merge subword tokens (e.g., 'dr' + '##owsy' -> 'drowsy'),
    reconstructing full entities and correcting their spans.
    Args:
        ner_results (list of dict): Output from the NER pipeline.
        text (str): The original input text.
    Returns:
        list of dict: Cleaned NER results with merged entities.
    """
    if not ner_results:
        return []
    merged_results = []
    prev = None
    for entity in ner_results:
        # If this is the first entity, just add it
        if prev is None:
            prev = entity.copy()
            continue
        # If the current entity is contiguous with the previous and has the same label, merge them
        if (entity['entity_group'] == prev['entity_group'] and entity['start'] == prev['end']):
            # Merge the words and update the end position
            prev['word'] += entity['word'].replace('##', '')
            prev['end'] = entity['end']
        else:
            # Finalize the previous entity
            # Re-extract the text from the original input to avoid tokenization artifacts
            prev['word'] = text[prev['start']:prev['end']]
            merged_results.append(prev)
            prev = entity.copy()
    # Add the last entity
    if prev is not None:
        prev['word'] = text[prev['start']:prev['end']]
        merged_results.append(prev)
    return merged_results

if __name__ == "__main__":
    # Example: process a single file for demonstration
    example_filename = "ARTHROTEC.1.txt"  # You can change this to any file in cadec/text
    text_dir = os.path.join("cadec", "text")
    # Step 1: Read the forum post
    forum_post = read_forum_post(text_dir, example_filename)
    print("--- Forum Post ---")
    print(forum_post)
    print("------------------")
    # Step 2: Run the model and postprocess the results
    ner_results = label_text_with_bio(forum_post)
    ner_results = postprocess_ner_results(ner_results, forum_post)
    print("\n--- BIO/IOB Labelling Results ---")
    for entity in ner_results:
        print(f"Text: '{entity['word']}' | Label: {entity['entity_group']} | Score: {entity['score']:.2f} | Start: {entity['start']} | End: {entity['end']}")
    print("---------------------------------")
    # Step 3: Convert to .ann format (after postprocessing and mapping)
    ann_lines = convert_ner_to_ann_format(ner_results)
    print("\n--- Converted to .ann Format ---")
    for line in ann_lines:
        print(line)
    # Write predictions to a file for evaluation in step 3
    with open("predicted.ann", "w", encoding="utf-8") as f:
        for line in ann_lines:
            f.write(line + "\n")
    print("\nPredictions written to predicted.ann for evaluation.")
    # Step 4: Print results (already done above) 