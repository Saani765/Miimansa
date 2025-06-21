# Step 3: Performance Evaluation of NER Labelling
# Assignment: Miimansa - Step 3
# Author: [Your Name]
# Date: [Today's Date]
#
# This script will:
#   - Compare the predicted .ann-style output from step 2 with the ground truth .ann files in cadec/original
#   - Measure performance using a suitable metric (e.g., precision, recall, F1-score)
#   - Justify the choice of metric in comments
#
# The code is written with beginners in mind, with clear comments and step-by-step structure.

import os
from collections import defaultdict

# 1. Function to read .ann files and parse entities
# 2. Function to compare predicted and ground truth entities
# 3. Function to compute precision, recall, F1-score
# 4. Main block to demonstrate evaluation on a single file

def read_ann_file(filepath):
    """
    Reads a .ann file and parses entities into a list of tuples.
    Each tuple contains: (label, start, end, text)
    Args:
        filepath (str): Path to the .ann file.
    Returns:
        list of tuples: (label, start, end, text)
    """
    entities = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip comments and empty lines
            parts = line.split('\t')
            if len(parts) < 3:
                continue  # Skip malformed lines
            # Example: T1\tDrug 44 49\tzocor
            label_ranges = parts[1]
            entity_text = parts[2]
            label_parts = label_ranges.split(' ')
            label = label_parts[0]
            # Some entities may have multiple ranges (e.g., 44 49;50 55), but for now, handle single range
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue  # Skip if indices are not valid
            entities.append((label, start, end, entity_text))
    return entities

def compare_entities(pred_entities, gt_entities):
    """
    Compares predicted and ground truth entities for exact matches.
    Args:
        pred_entities (list of tuples): Predicted entities (label, start, end, text)
        gt_entities (list of tuples): Ground truth entities (label, start, end, text)
    Returns:
        set: True positives (matched entities)
        set: False positives (predicted but not in ground truth)
        set: False negatives (ground truth but not predicted)
    """
    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    true_positives = pred_set & gt_set
    false_positives = pred_set - gt_set
    false_negatives = gt_set - pred_set
    return true_positives, false_positives, false_negatives

def compute_metrics(tp, fp, fn):
    """
    Computes precision, recall, and F1-score.
    Args:
        tp (set): True positives
        fp (set): False positives
        fn (set): False negatives
    Returns:
        dict: Precision, recall, and F1-score
    """
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Example: Evaluate predictions for a single file
    # TODO: Replace with actual file paths and predicted results from step 2
    pred_ann_file = "predicted.ann"  # Placeholder for predicted .ann file
    gt_ann_file = os.path.join("cadec", "original", "ARTHROTEC.1.ann")  # Example ground truth file
    # Example: Read and print entities from a ground truth .ann file
    print("--- Ground Truth Entities ---")
    gt_entities = read_ann_file(gt_ann_file)
    for ent in gt_entities:
        print(ent)
    # Example: Read predicted entities (replace with actual file in practice)
    print("\n--- Predicted Entities ---")
    if os.path.exists(pred_ann_file):
        pred_entities = read_ann_file(pred_ann_file)
        for ent in pred_entities:
            print(ent)
    else:
        pred_entities = []
        print("No predicted.ann file found. Please generate predictions from step 2.")
    # Compare and compute metrics
    tp, fp, fn = compare_entities(pred_entities, gt_entities)
    metrics = compute_metrics(tp, fp, fn)
    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1-score:  {metrics['f1']:.2f}")
    # Justification: F1-score is a balanced measure that considers both precision (how many predicted entities are correct) and recall (how many ground truth entities are found). It is widely used for NER evaluation. 