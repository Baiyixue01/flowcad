#!/usr/bin/env python3
"""
Script to count the number of lines (data entries) in a JSONL file.
Usage: python count_jsonl.py <filename>
"""

import sys
import json

def count_jsonl_entries(filename):
    """
    Count the number of entries in a JSONL file.
    Each line in a JSONL file is a separate JSON object.
    """
    count = 0
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    # Validate that it's valid JSON
                    json.loads(line)
                    count += 1
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON on line {count + 1}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    return count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_jsonl.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    num_entries = count_jsonl_entries(filename)
    print(f"Number of data entries in '{filename}': {num_entries}")