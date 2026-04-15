# src/utils/io.py
import json
import csv
import yaml
from pathlib import Path

def _json_default(o):
    """Convert NumPy/Pandas types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def read_json(path: Path) -> dict:
    """Read JSON file and return data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: Path, data: dict) -> None:
    """Write dictionary data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)

def read_csv(path: Path) -> list:
    """Read CSV file into a list of dictionaries."""
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def write_csv(path: Path, data: list, fieldnames: list) -> None:
    """Write list of dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def read_yaml(path: Path) -> dict:
    """Read YAML file and return data."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    import os
    os.makedirs(path, exist_ok=True)

def safe_read_csv(path, encodings=("utf-8", "utf-8-sig", "cp1252"), **kwargs):
    """
    Safely read CSV file with multiple encoding fallbacks.
    
    Uses open() with errors="replace" to handle decoding issues (compatible with older pandas).
    Tries encodings in order until one succeeds.
    
    This is essential for Windows where CSV files with Hindi/Punjabi text
    may have BOM markers or encoding issues.
    
    Args:
        path: Path to CSV file (Path object or string)
        encodings: Tuple of encodings to try in order (default: utf-8, utf-8-sig, cp1252)
        **kwargs: Additional arguments to pass to pd.read_csv (e.g., low_memory=False)
    
    Returns:
        Tuple of (DataFrame, encoding_used) where encoding_used is the encoding that succeeded
    
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If all encodings fail (raises the last error encountered)
    """
    import pandas as pd
    
    # Convert Path to string if needed
    path_str = str(path) if isinstance(path, Path) else path
    
    last_err = None
    for enc in encodings:
        try:
            # Use open() with errors="replace" instead of passing to pd.read_csv()
            # This works on older pandas versions that don't support errors= parameter
            with open(path_str, "r", encoding=enc, errors="replace", newline="") as f:
                df = pd.read_csv(f, **kwargs)
            
            # Normalize column names: strip whitespace
            df.columns = df.columns.astype(str).str.strip()
            
            return df, enc
        except Exception as e:
            last_err = e
            continue
    
    # If all encodings failed, raise the last error
    if last_err is None:
        raise ValueError(f"Could not read CSV file {path_str} with any encoding. Tried: {encodings}")
    raise last_err