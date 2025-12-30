"""
utils/file_io.py

File I/O utilities for loading and saving data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from config.settings import PATHS


def save_json(data: Any, filepath: Path = None, filename: str = None) -> Path:
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        filepath: Full path to file (overrides filename)
        filename: Filename (will use PATHS)
    
    Returns:
        Path where file was saved
    """
    if filepath is None:
        if filename is None:
            raise ValueError("Must provide either filepath or filename")
        filepath = PATHS.OUTPUT_DIR / filename
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_json(filepath: Path = None, filename: str = None) -> Any:
    """
    Load JSON file.
    
    Args:
        filepath: Full path to file (overrides filename)
        filename: Filename (will use PATHS)
    
    Returns:
        Loaded data
    """
    if filepath is None:
        if filename is None:
            raise ValueError("Must provide either filepath or filename")
        filepath = PATHS.OUTPUT_DIR / filename
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dataframe(
    df: pd.DataFrame, 
    filepath: Path = None, 
    filename: str = None,
    index: bool = False
) -> Path:
    """
    Save DataFrame as CSV.
    
    Args:
        df: DataFrame to save
        filepath: Full path to file (overrides filename)
        filename: Filename (will use PATHS)
        index: Whether to include index
    
    Returns:
        Path where file was saved
    """
    if filepath is None:
        if filename is None:
            raise ValueError("Must provide either filepath or filename")
        filepath = PATHS.OUTPUT_DIR / filename
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index)
    return filepath


def load_dataframe(
    filepath: Path = None, 
    filename: str = None
) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    
    Args:
        filepath: Full path to file (overrides filename)
        filename: Filename (will use PATHS)
    
    Returns:
        Loaded DataFrame
    """
    if filepath is None:
        if filename is None:
            raise ValueError("Must provide either filepath or filename")
        filepath = PATHS.INPUT_DIR / filename
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_csv(filepath)


def save_personas(personas: List[Dict], filepath: Path = None) -> Path:
    """
    Save personas to JSON file.
    
    Args:
        personas: List of persona dictionaries
        filepath: Path to save (default: PATHS.PERSONAS_JSON)
    
    Returns:
        Path where file was saved
    """
    if filepath is None:
        filepath = PATHS.PERSONAS_JSON
    
    return save_json(personas, filepath=filepath)


def load_personas(filepath: Path = None) -> List[Dict]:
    """
    Load personas from JSON file.
    
    Args:
        filepath: Path to load from (default: PATHS.PERSONAS_JSON)
    
    Returns:
        List of persona dictionaries
    """
    if filepath is None:
        filepath = PATHS.PERSONAS_JSON
    
    return load_json(filepath=filepath)


def check_file_exists(filepath: Path = None, filename: str = None) -> bool:
    """
    Check if a file exists.
    
    Args:
        filepath: Full path to file
        filename: Filename (will check in PATHS.OUTPUT_DIR)
    
    Returns:
        True if file exists
    """
    if filepath is None:
        if filename is None:
            return False
        filepath = PATHS.OUTPUT_DIR / filename
    
    return Path(filepath).exists()


def get_latest_file(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    """
    Get the most recently modified file in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (default: *.csv)
    
    Returns:
        Path to latest file, or None if no files found
    """
    files = list(Path(directory).glob(pattern))
    
    if not files:
        return None
    
    return max(files, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING FILE I/O UTILITIES")
    print("="*80 + "\n")
    
    # Test JSON save/load
    test_data = {"test": "data", "number": 42}
    
    print("Testing JSON save/load...")
    filepath = save_json(test_data, filename="test.json")
    print(f"✅ Saved to: {filepath}")
    
    loaded = load_json(filename="test.json")
    print(f"✅ Loaded: {loaded}")
    
    assert loaded == test_data, "Data mismatch!"
    print("✅ Data matches!")
    
    # Test DataFrame save/load
    print("\nTesting DataFrame save/load...")
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    filepath = save_dataframe(df, filename="test.csv")
    print(f"✅ Saved DataFrame to: {filepath}")
    
    loaded_df = load_dataframe(filename="test.csv")
    print(f"✅ Loaded DataFrame with shape: {loaded_df.shape}")
    
    # Clean up
    Path(filepath).unlink()
    (PATHS.OUTPUT_DIR / "test.json").unlink()
    print("\n✅ Cleanup complete")
    
    print("\n" + "="*80)
    print("✅ ALL FILE I/O TESTS PASSED")
    print("="*80 + "\n")