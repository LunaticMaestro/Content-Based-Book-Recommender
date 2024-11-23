# from z_utils import get_dataframe
# import numpy as np

# # CONST
# SUMMARY_VECTORS = "app_cache/summary_vectors.npy"
# BOOKS_CSV = "clean_books_summary.csv"

# def get_recommendation(book_title: str) -> str:
#     return book_title


# def sanity_check():
#     '''Validates whether the vectors count is of same as summaries present else RAISES Error
#     '''
#     global BOOKS_CSV, SUMMARY_VECTORS
#     df = get_dataframe(BOOKS_CSV)
#     vectors = np.load(SUMMARY_VECTORS)
#     assert df.shape[0] == vectors.shape[0]
    

# Reference: https://huggingface.co/learn/nlp-course/en/chapter9/2

import gradio as gr
from z_similarity import computes_similarity_w_hypothetical
from z_hypothetical_summary import generate_summaries

def get_recommendation(book_title: str):
    # Generate hypothetical summary
    fake_summaries = generate_summaries(book_title=book_title, n_samples=5) # other parameters are set to default in the function
    return fake_summaries[0]

# We instantiate the Textbox class
textbox = gr.Textbox(label="Write truth you wana Know:", placeholder="John Doe", lines=2)


demo = gr.Interface(fn=get_recommendation, inputs=textbox, outputs="text")

demo.launch()
