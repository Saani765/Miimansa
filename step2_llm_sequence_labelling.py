import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import namedtuple
import json

# --------- CONFIGURATION ---------
# You can change this to any file in cadec/text/
EXAMPLE_TEXT_FILE = 'cadec/text/ARTHROTEC.1.txt'

# Use a suitable NER model from Hugging Face (can be replaced with a more domain-specific model if available)
# MODEL_NAME = 'dslim/bert-base-NER'
MODEL_NAME = 'd4data/biomedical-ner-all'

# --------- LOAD MODEL ---------
print('Loading model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# --------- READ FORUM POST ---------
with open(EXAMPLE_TEXT_FILE, 'r', encoding='utf-8') as f:
    text = f.read().strip()
print(f'Loaded text from {EXAMPLE_TEXT_FILE}:\n{text[:200]}...\n')

# --------- a) BIO/IOB LABELLING ---------
print('Running NER pipeline...')
results = ner_pipeline(text)

# Mapping from model entity labels to assignment categories
entity_map = {
    'Drug': ['Drug'],
    'Disease_disorder': ['Disease'],
    'Sign_symptom': ['Symptom', 'ADR'],
    'Other_event': ['ADR'],
    'Detailed_description': ['ADR'],
    # Add more mappings if needed
}

# Build BIO tags for each word
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
labels = ['O'] * len(tokens)

# Map NER results to BIO tags, using only mapped categories
for ent in results:
    ent_tokens = tokenizer.tokenize(ent['word'])
    mapped_labels = entity_map.get(ent['entity_group'], None)
    if not mapped_labels:
        continue  # Skip entities not in our mapping
    for mapped_label in mapped_labels:
        # Find where these tokens appear in the full token list
        for i in range(len(tokens)):
            if tokens[i:i+len(ent_tokens)] == ent_tokens:
                # Only overwrite if current label is 'O' to avoid double-labeling
                if labels[i] == 'O':
                    labels[i] = 'B-' + mapped_label
                    for j in range(1, len(ent_tokens)):
                        if labels[i+j] == 'O':
                            labels[i+j] = 'I-' + mapped_label
                break

# Print tokens with BIO tags
print('Tokens with BIO tags:')
for token, label in zip(tokens, labels):
    print(f'{token}\t{label}')

# --------- b) CONVERT BIO TO SPAN FORMAT ---------
# Helper structure for spans
Span = namedtuple('Span', ['label', 'start', 'end', 'text'])

spans = []
current_label = None
start_idx = None
for i, label in enumerate(labels):
    if label.startswith('B-'):
        if current_label is not None:
            # End previous span
            end_idx = i
            span_text = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
            spans.append(Span(current_label, start_idx, end_idx, span_text))
        current_label = label[2:]
        start_idx = i
    elif label.startswith('I-'):
        continue
    else:  # 'O'
        if current_label is not None:
            end_idx = i
            span_text = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
            spans.append(Span(current_label, start_idx, end_idx, span_text))
            current_label = None
            start_idx = None
# Handle last span
if current_label is not None:
    end_idx = len(tokens)
    span_text = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
    spans.append(Span(current_label, start_idx, end_idx, span_text))

# Print span-format output
print('\nSpan-format output:')
for span in spans:
    print(f'Label: {span.label}, Tokens: {span.start}-{span.end}, Text: "{span.text}"')

# --------- SAVE SPANS TO JSON FOR STEP 3 ---------
# Save as list of [label, start, end, text]
predicted_spans = [[span.label, span.start, span.end, span.text] for span in spans]
output_json = os.path.basename(EXAMPLE_TEXT_FILE).replace('.txt', '_predicted_spans.json')
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(predicted_spans, f, ensure_ascii=False, indent=2)
print(f'Predicted spans saved to {output_json}')

# --------- NOTES ---------
# - The model used here is a general NER model. For best results, use a domain-specific model if available.
# - The mapping from model entity labels to ADR, Drug, Disease, Symptom is handled above.
# - This script demonstrates the process for a single file. You can loop over files for batch processing. 