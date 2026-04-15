# src/utils/split.py
from sklearn.model_selection import train_test_split

def stratified_split(X, y, test_size=0.1, val_size=0.1, random_state=42):
    """Perform stratified split for imbalanced datasets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    val_size_adjusted = val_size / (test_size + val_size)  # Adjust validation size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
