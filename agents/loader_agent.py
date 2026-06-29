import pandas as pd
from typing import Dict, Any, Union

class LoaderAgent:
    """
    Agent responsible for loading a dataset from a CSV file and extracting basic metadata.
    """
    def __init__(self):
        pass

    def load_data(self, file_path_or_buffer: Any) -> pd.DataFrame:
        """
        Loads a CSV dataset into a pandas DataFrame.
        Supports file paths or file-like buffers (from Streamlit's file uploader).
        Handles common encoding exceptions.
        """
        try:
            # Try reading with default UTF-8 encoding
            return pd.read_csv(file_path_or_buffer)
        except UnicodeDecodeError:
            # Fallback to Latin-1 if UTF-8 fails
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
            return pd.read_csv(file_path_or_buffer, encoding='latin-1')
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts high-level summary metadata from the loaded DataFrame.
        """
        return {
            "shape": df.shape,
            "rows": df.shape[0],
            "columns": list(df.columns),
            "num_columns": df.shape[1],
            "preview": df.head(5)
        }
