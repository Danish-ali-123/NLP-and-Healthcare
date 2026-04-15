import json
from collections import Counter

# Path to the JSONL file
jsonl_path = 'preprocess_jsonl/english_data.jsonl'

# Initialize counter for class distribution
class_counts = Counter()
total_records = 0

# Read and analyze the JSONL file
print(f"Analyzing class distribution in {jsonl_path}...")
print("=" * 60)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                label = data.get('label', 'unknown')
                class_counts[label] += 1
                total_records += 1
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON line: {line[:50]}...")
            except KeyError:
                print(f"⚠️  Missing 'label' field in line: {line[:50]}...")

# Print class distribution
print("Class Distribution:")
print("-" * 60)

if total_records > 0:
    # Print in descending order
    for label, count in class_counts.most_common():
        percentage = (count / total_records) * 100
        print(f"{label:30} | {count:8} | {percentage:6.2f}%")
    
    print("-" * 60)
    print(f"Total Records: {total_records}")
    print(f"Unique Classes: {len(class_counts)}")
    
    # Check for class imbalance
    most_common = class_counts.most_common()[0]
    least_common = class_counts.most_common()[-1]
    imbalance_ratio = most_common[1] / least_common[1]
    
    print(f"\nImbalance Analysis:")
    print(f"Most Common Class: {most_common[0]} ({most_common[1]} records)")
    print(f"Least Common Class: {least_common[0]} ({least_common[1]} records)")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 5:
        print("⚠️  WARNING: High class imbalance detected!")
    elif imbalance_ratio > 2:
        print("⚠️  Note: Moderate class imbalance detected")
    else:
        print("✅ Balanced class distribution")
else:
    print("❌ No valid records found in the file")
