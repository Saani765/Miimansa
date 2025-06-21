import os
import json
from step3_evaluate_predictions import read_ground_truth_spans, load_predicted_spans, normalize_span

# Load the list of sampled files
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
    gt_spans = read_ground_truth_spans(ann_file)
    predicted_spans = load_predicted_spans(pred_file)
    gt_spans_norm = set(normalize_span(s) for s in gt_spans)
    pred_spans_norm = set(normalize_span(s) for s in predicted_spans)
    true_positives = len(gt_spans_norm & pred_spans_norm)
    false_positives = len(pred_spans_norm - gt_spans_norm)
    false_negatives = len(gt_spans_norm - pred_spans_norm)
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

print(f"Evaluated {len(results)} posts. Skipped {skipped} due to missing files.")
print(f"Macro Precision: {avg_precision:.3f}")
print(f"Macro Recall:    {avg_recall:.3f}")
print(f"Macro F1-score:  {avg_f1:.3f}")

# Optionally, print per-file results
for r in results:
    print(f"{r['file']}: Precision={r['precision']:.3f}, Recall={r['recall']:.3f}, F1={r['f1']:.3f}") 