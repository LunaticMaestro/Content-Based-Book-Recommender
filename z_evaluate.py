# This is one time script, hence no functions but sequential coding

import random 
from z_utils import get_dataframe
from z_similarity import computes_similarity_w_hypothetical
from z_hypothetical_summary import generate_summaries
from tqdm import tqdm
import numpy as np

# CONST
random.seed(53)
CLEAN_DF_UNIQUE_TITLES = "unique_titles_books_summary.csv"
N_SAMPLES_EVAL = 30
TOP_K = 50
TOP_P = 0.85

books_df = get_dataframe(CLEAN_DF_UNIQUE_TITLES)

# sampling row id
random_values: list = random.sample(range(0, books_df.shape[0]), N_SAMPLES_EVAL)

reciprocal_ranks: list[int] = list() 

pbar = tqdm(total=N_SAMPLES_EVAL)

for idx in random_values:
    # Sample a book
    book = books_df.iloc[idx]

    # Generate hypothetical summary
    fake_summaries = generate_summaries(book_title = book["book_name"], n_samples=5, top_k=TOP_K, top_p=TOP_P)

    # Compute Simialrity 
    similarity, ranks = computes_similarity_w_hypothetical(hypothetical_summaries=fake_summaries)
    
    # Get reciprocal Rank
    df_ranked =  books_df.iloc[ranks]
    df_ranked = df_ranked.reset_index()
    df_ranked.drop(columns=["index"], inplace=True)
    rank = df_ranked[df_ranked["book_name"] == book["book_name"]].index.values[0] + 1 # rank starts 0 hence offseting by 1

    # Update list 
    reciprocal_ranks.append(1/rank)
    pbar.update(1) 

pbar.close()

print(f"USING Paramerters: TOP_K={TOP_K}  TOP_P={TOP_P}")
print("MRR: ", sum(reciprocal_ranks)/len(reciprocal_ranks))

# Calculate five-number summary
values = reciprocal_ranks
minimum = np.min(values)
q1 = np.percentile(values, 25)  # First quartile
median = np.median(values)
q3 = np.percentile(values, 75)  # Third quartile
maximum = np.max(values)

# Print the five-number summary
print("Five-Number Summary:")
print(f"Min: {minimum}")
print(f"Q1 : {q1}")
print(f"Med: {median}")
print(f"Q3 : {q3}")
print(f"Max: {maximum}")


