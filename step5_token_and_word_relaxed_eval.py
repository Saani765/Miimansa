import os
import json
import re
from step3_evaluate_predictions import load_predicted_spans

def read_ground_truth_spans_with_offsets(ann_file):
    spans = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            tag, label_ranges, entity_text = parts[0], parts[1], parts[2]
            label_parts = label_ranges.split(' ')
            label = label_parts[0]
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue
            spans.append((label, start, end, entity_text.strip()))
    return spans

def tokenize(text):
    # Simple whitespace and punctuation tokenizer
    return re.findall(r"\w+", text)

def token_level_pairs(spans):
    pairs = set()
    for label, start, end, text in spans:
        tokens = tokenize(text)
        for token in tokens:
            pairs.add((label.lower(), token.lower()))
    return pairs

def word_presence_match(pred_spans, gt_spans):
    # For each predicted entity, if any word in its span is present in any gold span of the same label, count as match
    gt_by_label = {}
    for label, start, end, text in gt_spans:
        gt_by_label.setdefault(label.lower(), []).append(set(tokenize(text)))
    pred_matched = set()
    gt_matched = set()
    for pi, (plabel, pstart, pend, ptext) in enumerate(pred_spans):
        plabel = plabel.lower()
        ptokens = set(tokenize(ptext))
        for gi, gt_tokens in enumerate(gt_by_label.get(plabel, [])):
            if ptokens & gt_tokens:
                pred_matched.add(pi)
                gt_matched.add((plabel, gi))
    return len(pred_matched), len(pred_spans) - len(pred_matched), sum(len(v) for v in gt_by_label.values()) - len(gt_matched)

with open('step5_sampled_files.txt', 'r') as f:
    sampled_txt_files = [line.strip() for line in f if line.strip()]

results_token = []
results_word = []
skipped = 0
for txt_file in sampled_txt_files:
    base = txt_file.replace('.txt', '')
    ann_file = os.path.join('cadec/original', base + '.ann')
    pred_file = base + '_predicted_spans.json'
    if not (os.path.exists(ann_file) and os.path.exists(pred_file)):
        skipped += 1
        continue
    gt_spans = read_ground_truth_spans_with_offsets(ann_file)
    predicted_spans = load_predicted_spans(pred_file)
    pred_spans = [(label, start, end, text) for (label, start, end, text) in predicted_spans]
    # Token-level F1
    gt_token_pairs = token_level_pairs(gt_spans)
    pred_token_pairs = token_level_pairs(pred_spans)
    tp = len(gt_token_pairs & pred_token_pairs)
    fp = len(pred_token_pairs - gt_token_pairs)
    fn = len(gt_token_pairs - pred_token_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    results_token.append({'file': txt_file, 'precision': precision, 'recall': recall, 'f1': f1})
    # Word-presence F1
    tp_w, fp_w, fn_w = word_presence_match(pred_spans, gt_spans)
    precision_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
    recall_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
    f1_w = 2 * precision_w * recall_w / (precision_w + recall_w) if (precision_w + recall_w) > 0 else 0.0
    results_word.append({'file': txt_file, 'precision': precision_w, 'recall': recall_w, 'f1': f1_w})

if results_token:
    avg_precision_token = sum(r['precision'] for r in results_token) / len(results_token)
    avg_recall_token = sum(r['recall'] for r in results_token) / len(results_token)
    avg_f1_token = sum(r['f1'] for r in results_token) / len(results_token)
else:
    avg_precision_token = avg_recall_token = avg_f1_token = 0.0

if results_word:
    avg_precision_word = sum(r['precision'] for r in results_word) / len(results_word)
    avg_recall_word = sum(r['recall'] for r in results_word) / len(results_word)
    avg_f1_word = sum(r['f1'] for r in results_word) / len(results_word)
else:
    avg_precision_word = avg_recall_word = avg_f1_word = 0.0

print(f"[TOKEN-LEVEL] Macro Precision: {avg_precision_token:.3f}")
print(f"[TOKEN-LEVEL] Macro Recall:    {avg_recall_token:.3f}")
print(f"[TOKEN-LEVEL] Macro F1-score:  {avg_f1_token:.3f}")
print(f"[WORD-PRESENCE] Macro Precision: {avg_precision_word:.3f}")
print(f"[WORD-PRESENCE] Macro Recall:    {avg_recall_word:.3f}")
print(f"[WORD-PRESENCE] Macro F1-score:  {avg_f1_word:.3f}") 