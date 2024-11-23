import pandas as pd
import numpy as np

def get_dataframe(file_path: str) -> pd.DataFrame:
    '''Reads a CSV file and returns a Pandas DataFrame.

    Args:
        file_path: The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    '''
    try:
        df = pd.read_csv(file_path)
        ## Minor tweak to fix the escape sequence character
        df['summaries'] = df['summaries'].str.replace('\xa0', '', regex=False)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_cache_embeddings(embedding_path: str="app_cache/summary_vectors.npy") -> np.ndarray:
    '''Returns embeddings of the book summaries'''
    emb = np.load(embedding_path)
    emb = emb.astype(np.float32) 
    return emb