import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.utils.io import write_json, ensure_dir, safe_read_csv
from src.config import base_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess multilingual clinical data for orthopedic diagnostics.

    Supports TWO input schemas:

    (A) Raw schema (original):
      - Pseudonymized_Patient History
      - Pseudonymized_Diagnosis Category
      - Pseudonymized_age
      - Pseudonymized_gender

    (B) Cleaned schema (model-ready):
      - text
      - label
      - age
      - gender
      - language (optional)
      - id (optional)
    """

    RAW_REQUIRED = [
        "Pseudonymized_Patient History",
        "Pseudonymized_Diagnosis Category",
        "Pseudonymized_age",
        "Pseudonymized_gender",
    ]

    CLEAN_REQUIRED = ["text", "label", "age", "gender"]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or base_config
        self.languages = self.config["data"]["languages"]
        
        # Schema candidates for auto-detection
        self.schema_candidates = [
            # Pseudonymized schema
            {
                "text": "Pseudonymized_Patient History",
                "label": "Pseudonymized_Diagnosis Category",
                "age": "Pseudonymized_age",
                "gender": "Pseudonymized_gender",
            },
            # Raw schema
            {
                "text": "Patient History",
                "label": "Diagnosis Category",
                "age": "age",
                "gender": "gender",
            },
        ]

    # -------------------------
    # File loading (robust)
    # -------------------------
    def load_raw_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        logger.info("Loading raw data files...")

        data_files = {
            "ENGLISH": os.path.join(data_dir, "ENGLISH.csv"),
            "HINDI": os.path.join(data_dir, "HINDI.csv"),
            "PUNJABI": os.path.join(data_dir, "PUNJABI.csv"),
        }

        dataframes = {}
        for lang, file_path in data_files.items():
            # Convert to absolute path and log it
            abs_path = os.path.abspath(file_path)
            logger.info(f"[{lang}] Attempting to load: {abs_path}")
            
            if not os.path.exists(abs_path):
                logger.error(f"[{lang}] ❌ File not found: {abs_path}")
                dataframes[lang] = pd.DataFrame()
                continue

            # Check file size
            file_size = os.path.getsize(abs_path)
            if file_size == 0:
                logger.error(f"[{lang}] ❌ File is empty (0 bytes): {abs_path}")
                dataframes[lang] = pd.DataFrame()
                continue

            logger.info(f"[{lang}] File exists, size: {file_size} bytes")

            # Use safe_read_csv helper (tries multiple encodings)
            try:
                df, used_enc = safe_read_csv(abs_path, low_memory=False)
                logger.info(f"[{lang}] ✅ Loaded with encoding={used_enc}, shape={df.shape}")
            except Exception as e:
                logger.error(f"[{lang}] ❌ Failed to read CSV: {str(e)}")
                dataframes[lang] = pd.DataFrame()
                continue

            # Normalize column names: strip whitespace, handle duplicates
            logger.info(f"[{lang}] Original columns ({len(df.columns)}): {list(df.columns)[:10]}")
            
            # Column names are already stripped by safe_read_csv, but handle duplicates
            
            # Handle duplicate columns (keep first, rename duplicates)
            if df.columns.duplicated().any():
                logger.warning(f"[{lang}] Found duplicate column names. Renaming duplicates...")
                seen = {}
                new_columns = []
                for col in df.columns:
                    if col in seen:
                        seen[col] += 1
                        new_columns.append(f"{col}_dup{seen[col]}")
                    else:
                        seen[col] = 0
                        new_columns.append(col)
                df.columns = new_columns
                logger.info(f"[{lang}] After deduplication: {list(df.columns)[:10]}")

            # Log final column list (first time or debug mode)
            logger.info(f"[{lang}] Final columns ({len(df.columns)}): {list(df.columns)}")
            
            # Check if DataFrame is empty
            if len(df) == 0:
                logger.error(f"[{lang}] ❌ Loaded 0 rows from file")
                logger.error(f"[{lang}] First 3 rows preview (if any):")
                logger.error(f"{df.head(3).to_string()}")
                dataframes[lang] = pd.DataFrame()
                continue

            dataframes[lang] = df
            logger.info(f"[{lang}] ✅ Loaded {len(df)} records successfully")

        return dataframes

    # -------------------------
    # Schema detection
    # -------------------------
    def detect_schema(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Detect which schema the dataframe uses by checking schema candidates."""
        if df is None or df.empty:
            return None
        
        # Normalize column names (strip whitespace) for comparison
        df_cols_normalized = {str(c).strip(): c for c in df.columns}
        
        # Try each schema candidate
        for schema in self.schema_candidates:
            # Check if all required columns exist (with whitespace tolerance)
            schema_cols_found = []
            for key, expected_col in schema.items():
                expected_col_stripped = expected_col.strip()
                found = False
                for df_col_stripped, orig_col in df_cols_normalized.items():
                    if df_col_stripped == expected_col_stripped:
                        schema_cols_found.append(orig_col)
                        found = True
                        break
                if not found:
                    break
            else:
                # All columns found for this schema
                return schema
        
        # Also check for clean schema (text, label, age, gender)
        clean_schema = {
            "text": "text",
            "label": "label",
            "age": "age",
            "gender": "gender",
        }
        clean_cols_found = []
        for key, expected_col in clean_schema.items():
            expected_col_stripped = expected_col.strip()
            found = False
            for df_col_stripped, orig_col in df_cols_normalized.items():
                if df_col_stripped == expected_col_stripped:
                    clean_cols_found.append(orig_col)
                    found = True
                    break
            if not found:
                break
        else:
            # All clean columns found
            return clean_schema
        
        return None

    def validate_data(self, df: pd.DataFrame, language: str) -> bool:
        if df is None or df.empty:
            logger.warning(f"Empty dataframe for {language}")
            return False

        schema = self.detect_schema(df)

        if schema is None:
            logger.error(f"[{language}] ❌ Unknown schema - required columns not found")
            logger.error(f"[{language}] Expected RAW columns: ['Patient History','Diagnosis Category','age','gender']")
            logger.error(f"[{language}] Expected CLEAN columns: ['text','label','age','gender']")
            logger.error(f"[{language}] Actual columns ({len(df.columns)}): {list(df.columns)}")
            logger.error(f"[{language}] First 3 rows preview:\n\n{df.head(3)}")
            return False

        # Store detected schema for later usage
        df.attrs["schema"] = schema
        
        # Basic null warnings
        # Find actual column names (with whitespace tolerance)
        df_cols_normalized = {str(c).strip(): c for c in df.columns}
        text_col = None
        label_col = None
        
        for key, expected_col in schema.items():
            expected_col_stripped = expected_col.strip()
            for df_col_stripped, orig_col in df_cols_normalized.items():
                if df_col_stripped == expected_col_stripped:
                    if key == "text":
                        text_col = orig_col
                    elif key == "label":
                        label_col = orig_col
                    break
        
        if text_col is None or label_col is None:
            logger.error(f"[{language}] ❌ Could not find required columns after schema detection")
            return False
        
        critical = [text_col, label_col]
        critical_nulls = df[critical].isnull().sum()
        if critical_nulls.any():
            logger.warning(f"[{language}] ⚠️ Null values in critical columns: {critical_nulls.to_dict()}")

        return True

    # -------------------------
    # Cleaning helpers
    # -------------------------
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = " ".join(text.split())
        return text

    def clean_labels(self, df: pd.DataFrame, language: str) -> pd.DataFrame:
        """
        Clean and normalize labels:
        1) Fix malformed Punjabi label with ORG_REPLACEMENT
        2) Merge Unknown/अगیاत into Other equivalents
        """
        df = df.copy()

        # Fix malformed Punjabi label
        if language.upper() == "PUNJABI":
            bad = "ਮਸੂਕਲੋਸਕੇਲਟORG_REPLACEMENTਿਕਾਰ"
            good = "ਮਸੂਕਲੋਸਕੇਲਟਲ ਵਿਕਾਰ"
            if bad in df["label"].astype(str).values:
                count = (df["label"].astype(str) == bad).sum()
                df.loc[df["label"].astype(str) == bad, "label"] = good
                logger.info(f"Fixed {count} instances of malformed label '{bad}' → '{good}' in {language}")

        # Merge Unknown → Other (language-specific)
        if language.upper() == "ENGLISH":
            if "Unknown" in df["label"].astype(str).values:
                count = (df["label"].astype(str) == "Unknown").sum()
                df.loc[df["label"].astype(str) == "Unknown", "label"] = "Other"
                logger.info(f"Merged {count} instances of 'Unknown' → 'Other' in {language}")

        if language.upper() == "HINDI":
            if "Unknown" in df["label"].astype(str).values:
                count = (df["label"].astype(str) == "Unknown").sum()
                df.loc[df["label"].astype(str) == "Unknown", "label"] = "अन्य"
                logger.info(f"Merged {count} instances of 'Unknown' → 'अन्य' (Other) in {language}")

        if language.upper() == "PUNJABI":
            # Some datasets use "Unknown" literally, some use "ਅਗਿਆਤ"
            for unk in ["Unknown", "ਅਗਿਆਤ"]:
                if unk in df["label"].astype(str).values:
                    count = (df["label"].astype(str) == unk).sum()
                    df.loc[df["label"].astype(str) == unk, "label"] = "ਹੋਰ"
                    logger.info(f"Merged {count} instances of '{unk}' → 'ਹੋਰ' (Other) in {language}")

        return df

    # -------------------------
    # Main per-language preparation
    # -------------------------
    def prepare_single_language_data(self, df: pd.DataFrame, language: str) -> pd.DataFrame:
        if not self.validate_data(df, language):
            logger.error(f"[{language}] ❌ Validation failed, skipping")
            return pd.DataFrame()

        # Get schema from df.attrs (stored by validate_data) or detect it
        schema = df.attrs.get("schema") or self.detect_schema(df)
        
        if schema is None:
            logger.error(f"[{language}] ❌ Could not detect schema")
            return pd.DataFrame()
        
        logger.info(f"[{language}] Preparing data... (detected schema with columns: {list(schema.values())})")

        lang_map = {"ENGLISH": "en", "HINDI": "hi", "PUNJABI": "pa"}
        lang_code = lang_map.get(language.upper(), language.lower())

        # Check if this is already a clean schema (text, label, age, gender)
        if schema.get("text") == "text" and schema.get("label") == "label":
            # Already cleaned/model-ready
            processed_df = df.copy()

            # Normalize column names (strip whitespace)
            processed_df.columns = processed_df.columns.astype(str).str.strip()

            # Ensure standard columns exist
            missing_cols = []
            for col in ["text", "label", "age", "gender"]:
                if col not in processed_df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                logger.error(f"[{language}] ❌ Missing required clean columns: {missing_cols}")
                logger.error(f"[{language}] Available columns: {list(processed_df.columns)}")
                return pd.DataFrame()

            # Ensure language/id exist
            if "language" not in processed_df.columns:
                processed_df["language"] = lang_code
            if "id" not in processed_df.columns:
                processed_df["id"] = [f"{lang_code}_{i}" for i in range(len(processed_df))]

            # Keep only standard columns (ignore extra columns safely)
            keep_cols = ["text", "label", "age", "gender", "language", "id"]
            processed_df = processed_df[keep_cols]
        else:
            # Raw schema (pseudonymized or non-pseudonymized) - need to map columns
            # Find actual column names (handle whitespace variations)
            df_cols_normalized = {str(c).strip(): c for c in df.columns}
            
            selected_cols = []
            rename_mapping = {}
            
            for key, expected_col in schema.items():
                expected_col_stripped = expected_col.strip()
                found_col = None
                
                # Try exact match first
                if expected_col in df.columns:
                    found_col = expected_col
                else:
                    # Try normalized match
                    for df_col_stripped, orig_col in df_cols_normalized.items():
                        if df_col_stripped == expected_col_stripped:
                            found_col = orig_col
                            break
                
                if found_col is None:
                    logger.error(f"[{language}] ❌ Required column not found: '{expected_col}'")
                    logger.error(f"[{language}] Available columns: {list(df.columns)}")
                    return pd.DataFrame()
                
                selected_cols.append(found_col)
                # Map to standard output column name
                rename_mapping[found_col] = key

            # Select and rename columns
            processed_df = df[selected_cols].copy()
            processed_df = processed_df.rename(columns=rename_mapping)

        # Clean text + labels
        processed_df["text"] = processed_df["text"].apply(self.clean_text)
        processed_df = self.clean_labels(processed_df, language)

        # Force correct language code (overrides wrong user content)
        processed_df["language"] = lang_code

        # Ensure IDs are consistent (keep existing if already there)
        if "id" not in processed_df.columns:
            processed_df["id"] = [f"{language.lower()}_{i}" for i in range(len(processed_df))]

        # Drop invalid rows (missing label/text)
        initial_count = len(processed_df)

        missing_label = processed_df["label"].isna() | (processed_df["label"].astype(str).str.strip() == "")
        missing_text = processed_df["text"].isna() | (processed_df["text"].astype(str).str.strip() == "") | (
            processed_df["text"].astype(str).str.len() == 0
        )

        dropped_label_count = int(missing_label.sum())
        dropped_text_count = int(missing_text.sum())

        processed_df = processed_df[~missing_label & ~missing_text].copy()

        final_count = len(processed_df)

        logger.info(f"[{language}] Data cleaning summary:")
        logger.info(f"  Total rows loaded: {initial_count}")
        logger.info(f"  Dropped (missing label): {dropped_label_count}")
        logger.info(f"  Dropped (missing text): {dropped_text_count}")
        logger.info(f"  Final rows kept: {final_count}")

        if final_count == 0:
            logger.warning(f"{language}: after cleaning, no rows remain.")

        return processed_df

    # -------------------------
    # Combine + split + save
    # -------------------------
    def combine_all_data(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        combined_data = []
        failed_languages = []

        for language, df in dataframes.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {language}")
            logger.info(f"{'='*60}")
            
            if df is None or df.empty:
                logger.error(f"[{language}] ❌ Input dataframe is empty or None")
                failed_languages.append(f"{language} (empty input)")
                continue

            processed_df = self.prepare_single_language_data(df, language)
            
            if processed_df.empty:
                logger.error(f"[{language}] ❌ Processing resulted in empty dataframe")
                failed_languages.append(f"{language} (processing failed)")
                continue

            combined_data.append(processed_df)
            logger.info(f"[{language}] ✅ Successfully processed {len(processed_df)} records")

        if not combined_data:
            error_msg = "❌ No valid data found for any language.\n"
            error_msg += "Failed languages:\n"
            for failed in failed_languages:
                error_msg += f"  - {failed}\n"
            error_msg += "\nPlease check:\n"
            error_msg += "  1. File paths are correct\n"
            error_msg += "  2. Files contain required columns\n"
            error_msg += "  3. Files are not empty\n"
            error_msg += "  4. Encoding is UTF-8 compatible\n"
            logger.error(error_msg)
            raise ValueError(error_msg)

        combined_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Combined total {len(combined_df)} records from {len(combined_data)} languages")
        logger.info(f"{'='*60}")
        return combined_df

    def get_label_mapping(self, df: pd.DataFrame) -> Dict[str, Any]:
        unique_labels = sorted(df["label"].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        label_info = {
            "labels": unique_labels,
            "label2id": label2id,
            "id2label": id2label,
            "num_labels": len(unique_labels),
        }

        logger.info(f"Created label mapping with {len(unique_labels)} unique labels")
        logger.info(f"Labels: {unique_labels}")

        return label_info

    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        default_split = {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_state": self.config.get("seed", 42),
        }

        data_config = self.config.get("data", {})
        split_config = data_config.get("split", {})

        split_config = {
            "train_ratio": split_config.get("train_ratio", default_split["train_ratio"]),
            "val_ratio": split_config.get("val_ratio", default_split["val_ratio"]),
            "test_ratio": split_config.get("test_ratio", default_split["test_ratio"]),
            "random_state": split_config.get("random_state", default_split["random_state"]),
        }

        if "split" not in data_config:
            logger.warning("Split config not found in config, using defaults: train=0.8, val=0.1, test=0.1")

        # Check for patient_id or id column for group-based splitting (prevents leakage)
        use_group_split = False
        group_column = None
        
        # Check for 'id' or 'patient_id' column
        if 'id' in df.columns:
            group_column = 'id'
            # Check if same ID appears multiple times (indicates need for group split)
            id_counts = df[group_column].value_counts()
            if (id_counts > 1).any():
                use_group_split = True
                logger.info(f"Detected multiple records per ID. Using GROUP split by '{group_column}' to prevent leakage.")
                logger.info(f"  Unique IDs: {df[group_column].nunique()}, Total records: {len(df)}")
                logger.info(f"  IDs with multiple records: {(id_counts > 1).sum()}")
        elif 'patient_id' in df.columns:
            group_column = 'patient_id'
            id_counts = df[group_column].value_counts()
            if (id_counts > 1).any():
                use_group_split = True
                logger.info(f"Detected multiple records per patient_id. Using GROUP split by '{group_column}' to prevent leakage.")
                logger.info(f"  Unique patient_ids: {df[group_column].nunique()}, Total records: {len(df)}")
                logger.info(f"  Patient IDs with multiple records: {(id_counts > 1).sum()}")
        
        df_stratify = df["label"].astype(str) + "_" + df["language"].astype(str)

        stratify_counts = df_stratify.value_counts()
        min_samples_per_stratum = 2
        problematic = stratify_counts[stratify_counts < min_samples_per_stratum]

        if len(problematic) > 0:
            logger.warning(
                f"Found {len(problematic)} strata with < {min_samples_per_stratum} samples. Falling back to label-only stratification."
            )
            stratify_key = df["label"]
        else:
            stratify_key = df_stratify
            logger.info("Using stratified splitting by label AND language")

        # Use group-based split if patient_id/id detected and has duplicates
        if use_group_split and group_column:
            try:
                # Group-based split: ensure same patient never appears in multiple splits
                groups = df[group_column].values
                
                # First split: train vs (val+test)
                gss1 = GroupShuffleSplit(
                    n_splits=1,
                    test_size=split_config["val_ratio"] + split_config["test_ratio"],
                    random_state=split_config["random_state"]
                )
                train_idx, temp_idx = next(gss1.split(df, groups=groups))
                train_df = df.iloc[train_idx].copy()
                temp_df = df.iloc[temp_idx].copy()
                
                logger.info(f"Group split (train vs temp): train={len(train_df)}, temp={len(temp_df)}")
                
                # Second split: val vs test (within temp)
                val_ratio_adjusted = split_config["val_ratio"] / (split_config["val_ratio"] + split_config["test_ratio"])
                groups_temp = temp_df[group_column].values
                
                gss2 = GroupShuffleSplit(
                    n_splits=1,
                    test_size=1 - val_ratio_adjusted,
                    random_state=split_config["random_state"]
                )
                val_idx, test_idx = next(gss2.split(temp_df, groups=groups_temp))
                val_df = temp_df.iloc[val_idx].copy()
                test_df = temp_df.iloc[test_idx].copy()
                
                logger.info(f"Group split (val vs test): val={len(val_df)}, test={len(test_df)}")
                
                # Verify no leakage
                train_ids = set(train_df[group_column].unique())
                val_ids = set(val_df[group_column].unique())
                test_ids = set(test_df[group_column].unique())
                
                if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
                    logger.warning("⚠️ WARNING: Leakage detected after group split! Same IDs appear in multiple splits.")
                else:
                    logger.info("✓ No leakage: All IDs are unique across splits")
                    
            except Exception as e:
                logger.warning(f"Group split failed: {e}. Falling back to standard stratified split.")
                use_group_split = False
        
        # Standard stratified split (fallback or if no group column)
        if not use_group_split:
            try:
                train_df, temp_df = train_test_split(
                    df,
                    test_size=split_config["val_ratio"] + split_config["test_ratio"],
                    random_state=split_config["random_state"],
                    stratify=stratify_key,
                )
            except ValueError as e:
                logger.warning(f"Stratification failed: {e}. Falling back to label-only.")
                train_df, temp_df = train_test_split(
                    df,
                    test_size=split_config["val_ratio"] + split_config["test_ratio"],
                    random_state=split_config["random_state"],
                    stratify=df["label"],
                )

            # Only do second split if not using group split (group split already did it)
            if not use_group_split:
                val_ratio_adjusted = split_config["val_ratio"] / (split_config["val_ratio"] + split_config["test_ratio"])

                temp_stratify = temp_df["label"].astype(str) + "_" + temp_df["language"].astype(str)
                temp_counts = temp_stratify.value_counts()
                temp_problematic = temp_counts[temp_counts < min_samples_per_stratum]

                temp_stratify_key = temp_df["label"] if len(temp_problematic) > 0 else temp_stratify

                try:
                    val_df, test_df = train_test_split(
                        temp_df,
                        test_size=1 - val_ratio_adjusted,
                        random_state=split_config["random_state"],
                        stratify=temp_stratify_key,
                    )
                except ValueError as e:
                    logger.warning(f"Val/test stratification failed: {e}. Falling back to label-only.")
                    val_df, test_df = train_test_split(
                        temp_df,
                        test_size=1 - val_ratio_adjusted,
                        random_state=split_config["random_state"],
                        stratify=temp_df["label"],
                    )

        splits = {"train": train_df, "val": val_df, "test": test_df}

        logger.info("Data split completed:")
        for split_name, split_df in splits.items():
            logger.info(f"  {split_name}: {len(split_df)} records ({len(split_df)/len(df)*100:.1f}%)")
            logger.info(f"    Labels: {split_df['label'].nunique()} unique, Languages: {split_df['language'].nunique()} unique")

        return splits

    def calculate_statistics(self, splits: Dict[str, pd.DataFrame], label_info: Dict[str, Any]) -> Dict[str, Any]:
        stats = {}

        for split_name, split_df in splits.items():
            split_stats = {
                "total_records": len(split_df),
                "language_distribution": split_df["language"].value_counts().to_dict(),
                "label_distribution": split_df["label"].value_counts().to_dict(),
                "text_length_stats": {
                    "mean": float(split_df["text"].str.len().mean()) if len(split_df) > 0 else 0.0,
                    "std": float(split_df["text"].str.len().std()) if len(split_df) > 0 else 0.0,
                    "min": int(split_df["text"].str.len().min()) if len(split_df) > 0 else 0,
                    "max": int(split_df["text"].str.len().max()) if len(split_df) > 0 else 0,
                },
                "per_language": {},
            }

            for lang in split_df["language"].unique():
                lang_df = split_df[split_df["language"] == lang]
                split_stats["per_language"][lang] = {
                    "total_records": len(lang_df),
                    "unique_labels": int(lang_df["label"].nunique()),
                    "label_distribution": lang_df["label"].value_counts().to_dict(),
                    "text_length_stats": {
                        "mean": float(lang_df["text"].str.len().mean()) if len(lang_df) > 0 else 0.0,
                        "std": float(lang_df["text"].str.len().std()) if len(lang_df) > 0 else 0.0,
                        "min": int(lang_df["text"].str.len().min()) if len(lang_df) > 0 else 0,
                        "max": int(lang_df["text"].str.len().max()) if len(lang_df) > 0 else 0,
                    },
                }

            stats[split_name] = split_stats

        all_data = pd.concat(splits.values(), ignore_index=True)
        stats["overall"] = {
            "total_records": len(all_data),
            "num_languages": len(all_data["language"].unique()),
            "num_labels": len(label_info["labels"]),
            "languages": list(all_data["language"].unique()),
            "labels": label_info["labels"],
            "per_language_label_counts": {
                lang: int(all_data[all_data["language"] == lang]["label"].nunique())
                for lang in all_data["language"].unique()
            },
        }

        return stats

    def save_processed_data(self, splits: Dict[str, pd.DataFrame], label_info: Dict[str, Any], output_dir: str):
        ensure_dir(output_dir)

        for split_name, split_df in splits.items():
            output_path = os.path.join(output_dir, f"{split_name}.csv")
            split_df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"Saved {split_name} data to {output_path}")

        labels_path = os.path.join(output_dir, "labels.json")
        write_json(Path(labels_path), label_info)
        logger.info(f"Saved label information to {labels_path}")

        stats = self.calculate_statistics(splits, label_info)
        stats_path = os.path.join(output_dir, "statistics.json")
        write_json(Path(stats_path), stats)
        logger.info(f"Saved statistics to {stats_path}")

    def run_pipeline(self, raw_data_dir: str, output_dir: str) -> Dict[str, Any]:
        logger.info("Starting data preprocessing pipeline...")

        raw_dataframes = self.load_raw_data(raw_data_dir)
        combined_df = self.combine_all_data(raw_dataframes)
        label_info = self.get_label_mapping(combined_df)
        splits = self.split_data(combined_df)
        self.save_processed_data(splits, label_info, output_dir)

        logger.info("Data preprocessing pipeline completed successfully!")

        return {"splits": splits, "label_info": label_info, "statistics": self.calculate_statistics(splits, label_info)}


def main():
    preprocessor = DataPreprocessor()

    raw_data_dir = base_config["paths"]["data_raw"]
    processed_data_dir = base_config["paths"]["data_processed"]

    try:
        preprocessor.run_pipeline(raw_data_dir, processed_data_dir)
        logger.info("Data preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
