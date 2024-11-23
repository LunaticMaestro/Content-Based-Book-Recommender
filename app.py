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


def greet(name):
    return "Hello " + name


# We instantiate the Textbox class
textbox = gr.Textbox(label="Write truth you wana Know:", placeholder="John Doe", lines=2)


demo = gr.Interface(fn=greet, inputs=textbox, outputs="text")

demo.launch()
