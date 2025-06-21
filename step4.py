# Step 4: Performance Evaluation for ADR Label Only (Ground Truth: meddra)
# Assignment: Miimansa - Step 4
# Author: [Your Name]
# Date: [Today's Date]
#
# This script will:
#   - Compare the predicted .ann-style output from step 2 with the ground truth .ann files in cadec/meddra
#   - Treat all entities in meddra as ADRs
#   - Measure performance using precision, recall, F1-score
#   - Justify the choice of metric in comments

import os
from collections import defaultdict

def read_ann_file_all_as_adr(filepath):
    """
    Reads a .ann file and parses all entities as ADRs into a list of tuples.
    Each tuple contains: (label, start, end, text), where label is always 'ADR'.
    Args:
        filepath (str): Path to the .ann file.
    Returns:
        list of tuples: (label, start, end, text) for all entities, label set to 'ADR'
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
            label_ranges = parts[1]
            entity_text = parts[2]
            label_parts = label_ranges.split(' ')
            # Ignore the original label/code, treat all as 'ADR'
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue  # Skip if indices are not valid
            entities.append(('ADR', start, end, entity_text))
    return entities

def read_ann_file_adr_only(filepath):
    """
    Reads a .ann file and parses only 'ADR' entities into a list of tuples.
    Each tuple contains: (label, start, end, text)
    Args:
        filepath (str): Path to the .ann file.
    Returns:
        list of tuples: (label, start, end, text) for ADR only
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
            label_ranges = parts[1]
            entity_text = parts[2]
            label_parts = label_ranges.split(' ')
            label = label_parts[0]
            if label != 'ADR':
                continue  # Only keep ADR entities
            try:
                start = int(label_parts[1])
                end = int(label_parts[2])
            except (IndexError, ValueError):
                continue  # Skip if indices are not valid
            entities.append((label, start, end, entity_text))
    return entities

def compare_entities(pred_entities, gt_entities):
    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    true_positives = pred_set & gt_set
    false_positives = pred_set - gt_set
    false_negatives = gt_set - pred_set
    return true_positives, false_positives, false_negatives

def compute_metrics(tp, fp, fn):
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Example: Evaluate predictions for a single file (ground truth: all meddra entities as ADR, predictions: only ADR)
    pred_ann_file = "predicted.ann"  # Placeholder for predicted .ann file
    gt_ann_file = os.path.join("cadec", "meddra", "ARTHROTEC.1.ann")  # Example ground truth file
    print("--- Ground Truth ADR Entities (meddra, all treated as ADR) ---")
    gt_entities = read_ann_file_all_as_adr(gt_ann_file)
    for ent in gt_entities:
        print(ent)
    print("\n--- Predicted ADR Entities (only ADR label) ---")
    if os.path.exists(pred_ann_file):
        pred_entities = read_ann_file_adr_only(pred_ann_file)
        for ent in pred_entities:
            print(ent)
    else:
        pred_entities = []
        print("No predicted.ann file found. Please generate predictions from step 2.")
    tp, fp, fn = compare_entities(pred_entities, gt_entities)
    metrics = compute_metrics(tp, fp, fn)
    print("\n--- Evaluation Metrics (ADR only, all meddra entities) ---")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1-score:  {metrics['f1']:.2f}")
    # Justification: F1-score is a balanced measure that considers both precision (how many predicted ADR entities are correct) and recall (how many ground truth ADR entities are found). It is widely used for NER evaluation, especially for specific entity types like ADR. 