import os
import sys
import json
from collections import Counter

def read_ground_truth_spans(ann_file):
    """Read ground truth spans from a .ann file in cadec/original/."""
    spans = set()
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            tag, label_ranges, entity_text = parts[0], parts[1], parts[2]
            label = label_ranges.split(' ')[0]
            # Store as (label, text) for comparison
            spans.add((label, entity_text.strip()))
    return spans

def load_predicted_spans(json_file):
    """Load predicted spans from a JSON file. Expects list of [label, start, end, text] or [label, text]."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    spans = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # dict format: {"label": ..., "start": ..., "end": ..., "text": ...}
                spans.append((item['label'], item.get('start', 0), item.get('end', 0), item['text']))
            elif isinstance(item, list) or isinstance(item, tuple):
                if len(item) == 4:
                    # [label, start, end, text]
                    spans.append(tuple(item))
                elif len(item) == 2:
                    # [label, text]
                    spans.append((item[0], None, None, item[1]))
    return spans

def normalize_span(span):
    # Normalize label and text: lowercase, strip whitespace, collapse multiple spaces
    label, text = span[0], span[-1]
    return (label.strip().lower(), ' '.join(text.strip().lower().split()))

# For demonstration, load predictions from a file if you saved them, or paste them above
# Example usage: python step3_evaluate_predictions.py ARTHROTEC.1.ann
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python step3_evaluate_predictions.py <filename.ann> <predicted_spans.json>')
        print('Predicted spans JSON should be a list of [label, start, end, text] as output by the step2 script.')
        sys.exit(1)
    ann_filename = sys.argv[1]
    pred_json = sys.argv[2]
    ann_path = os.path.join('cadec/original', ann_filename)
    gt_spans = read_ground_truth_spans(ann_path)
    predicted_spans = load_predicted_spans(pred_json)
    # Convert predicted spans to (label, text) for comparison
    pred_spans = set((label, text.strip()) for (label, _, _, text) in predicted_spans)

    print('Ground Truth Spans:')
    for span in sorted(gt_spans):
        print(span)
    print('\nPredicted Spans:')
    for span in sorted(pred_spans):
        print(span)

    # Optionally normalize for more forgiving matching
    normalize = True
    if normalize:
        gt_spans_norm = set(normalize_span(s) for s in gt_spans)
        pred_spans_norm = set(normalize_span(s) for s in predicted_spans)
    else:
        gt_spans_norm = gt_spans
        pred_spans_norm = pred_spans

    # Calculate metrics
    true_positives = len(gt_spans_norm & pred_spans_norm)
    false_positives = len(pred_spans_norm - gt_spans_norm)
    false_negatives = len(gt_spans_norm - pred_spans_norm)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print('\n--- Normalized Matching Results ---')
    print(f'Precision: {precision:.3f}')
    print(f'Recall:    {recall:.3f}')
    print(f'F1-score:  {f1:.3f}')
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'False Negatives: {false_negatives}')
    print('\nJustification:')
    print('We use span-level (entity-level) precision, recall, and F1-score, which is standard for NER evaluation. This metric rewards exact matches of both label and text, and is robust to partial overlaps or tokenization differences. Here, we also show results with normalization (lowercase, whitespace normalization) to account for minor formatting differences.') 