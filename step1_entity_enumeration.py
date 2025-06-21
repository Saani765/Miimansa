import os
from collections import defaultdict

# Path to the 'original' annotation files
directory = 'cadec/original'

# Dictionary to store unique entities for each label
distinct_entities = defaultdict(set)

# Iterate through all annotation files in the directory
for filename in os.listdir(directory):
    if not filename.endswith('.ann'):
        continue
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip comments and empty lines
            parts = line.split('\t')
            if len(parts) < 3:
                continue  # Skip malformed lines
            tag, label_ranges, entity_text = parts[0], parts[1], parts[2]
            label = label_ranges.split(' ')[0]  # Label is the first word in the second column
            distinct_entities[label].add(entity_text)

# Print the results
for label in ['ADR', 'Drug', 'Disease', 'Symptom']:
    entities = distinct_entities[label]
    print(f'Label: {label}')
    print(f'Total unique entities: {len(entities)}')
    print(f'Example entities: {list(entities)[:10]}')
    print('-' * 40) 