import os
import json
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

def overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)

with open('step5_sampled_files.txt', 'r') as f:
    sampled_txt_files = [line.strip() for line in f if line.strip()]

results = []
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
    # Convert to (label, start, end, text)
    pred_spans = [(label, start, end, text) for (label, start, end, text) in predicted_spans]
    gt_matched = set()
    pred_matched = set()
    # Relaxed matching: overlap in span and same label
    for pi, (plabel, pstart, pend, ptext) in enumerate(pred_spans):
        for gi, (glabel, gstart, gend, gtext) in enumerate(gt_spans):
            if plabel.lower() == glabel.lower() and overlap(pstart, pend, gstart, gend):
                gt_matched.add(gi)
                pred_matched.add(pi)
    true_positives = len(pred_matched)
    false_positives = len(pred_spans) - len(pred_matched)
    false_negatives = len(gt_spans) - len(gt_matched)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    results.append({'file': txt_file, 'precision': precision, 'recall': recall, 'f1': f1})

if results:
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
else:
    avg_precision = avg_recall = avg_f1 = 0.0

print(f"[RELAXED] Evaluated {len(results)} posts. Skipped {skipped} due to missing files.")
print(f"[RELAXED] Macro Precision: {avg_precision:.3f}")
print(f"[RELAXED] Macro Recall:    {avg_recall:.3f}")
print(f"[RELAXED] Macro F1-score:  {avg_f1:.3f}")

for r in results:
    print(f"{r['file']}: Precision={r['precision']:.3f}, Recall={r['recall']:.3f}, F1={r['f1']:.3f}") 