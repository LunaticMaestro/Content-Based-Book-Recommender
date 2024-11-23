from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from z_utils import get_dataframe
from tqdm import tqdm 

# CONST
EMB_MODEL = "all-MiniLM-L6-v2"
INP_DATASET_CSV = "unique_titles_books_summary.csv" 
CACHE_SUMMARY_EMB_NPY = "app_cache/summary_vectors.npy"

# Load Model
#   setting this at global level because entire runtime will continue to use this model.

import gradio as gr

if gr.NO_RELOAD:
	model = SentenceTransformer(EMB_MODEL)



def dataframe_compute_summary_vector(books_df: pd.DataFrame) -> np.ndarray:
    '''Takes books summaries and compute embedding vectors 

    WARNING: Generated output will return the order of the dataframe; so yeah dont think about DELTA changes.

    Args:
        books_df: The input DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with new column `vector`
    '''
    global model
    
    if 'summaries' not in books_df.columns:
        raise ValueError("DataFrame must contain 'summaries' columns.")

    # Progress bar
    pbar = tqdm(total=books_df.shape[0])

    def encode(text:str): 
        pbar.update(1)
        try: 
            return model.encode(text)
        except TypeError as t: # Handle NAN
            return np.zeros(384) # HARDCODED
    

    summary_vectors = books_df['summaries'].map(encode).to_numpy()

    pbar.close()

    # reorder array size (N_ROWS, 384) 
    summary_vectors = np.stack(summary_vectors)

    return summary_vectors

def get_embeddings(summaries: list[str]) -> np.ndarray: 
    '''Utils function to to take in hypothetical document(s) and return the embedding of it(s)
    '''
    global model
    if isinstance(summaries, str): 
        summaries = [summaries, ]
    return model.encode(summaries)


def cache_create_embeddings(books_csv_path: str, output_path: str) -> None:
    '''Read the books csv and generate vectors of the `summaries` columns and store in `output_path` 
    '''
    books_df = get_dataframe(books_csv_path)
    vectors = dataframe_compute_summary_vector(books_df)
    np.save(file=output_path, arr=vectors)
    print(f"Vectors saved to {output_path}")


if __name__ == "__main__":
    print("Generating vectors of the summaries")
    cache_create_embeddings(books_csv_path=INP_DATASET_CSV, output_path=CACHE_SUMMARY_EMB_NPY)