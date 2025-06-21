import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = 'd4data/biomedical-ner-all'

print('Loading model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Helper: postprocess NER results to merge subword tokens

def postprocess_ner_results(ner_results, text):
    if not ner_results:
        return []
    merged_results = []
    prev = None
    for entity in ner_results:
        if prev is None:
            prev = entity.copy()
            continue
        if (entity['entity_group'] == prev['entity_group'] and entity['start'] == prev['end']):
            prev['word'] += entity['word'].replace('##', '')
            prev['end'] = entity['end']
        else:
            prev['word'] = text[prev['start']:prev['end']]
            merged_results.append(prev)
            prev = entity.copy()
    if prev is not None:
        prev['word'] = text[prev['start']:prev['end']]
        merged_results.append(prev)
    return merged_results

# Helper: map model entity labels to assignment categories
entity_map = {
    'Drug': ['Drug'],
    'Disease_disorder': ['Disease'],
    'Sign_symptom': ['Symptom', 'ADR'],
    'Other_event': ['ADR'],
    'Detailed_description': ['ADR'],
}

def get_mapped_labels(entity_group):
    return entity_map.get(entity_group, None)

# Main batch loop
with open('step5_sampled_files.txt', 'r') as f:
    sampled_txt_files = [line.strip() for line in f if line.strip()]

for txt_file in sampled_txt_files:
    text_path = os.path.join('cadec/text', txt_file)
    if not os.path.exists(text_path):
        print(f"Text file missing: {txt_file}")
        continue
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"Processing {txt_file} ...")
    ner_results = ner_pipeline(text)
    ner_results = postprocess_ner_results(ner_results, text)
    # Convert to span format: [label, start, end, text]
    predicted_spans = []
    for entity in ner_results:
        mapped_labels = get_mapped_labels(entity['entity_group'])
        if not mapped_labels:
            continue
        for mapped_label in mapped_labels:
            predicted_spans.append([mapped_label, entity['start'], entity['end'], entity['word']])
    # Save to JSON
    base = txt_file.replace('.txt', '')
    out_json = f"{base}_predicted_spans.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(predicted_spans, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_json}") 