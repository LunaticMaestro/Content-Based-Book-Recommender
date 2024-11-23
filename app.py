# CONST
CLEAN_DF_UNIQUE_TITLES = "unique_titles_books_summary.csv"
N_RECOMMENDS = 5

# def get_recommendation(book_title: str) -> str:
#     return book_title

# from transformers import pipeline, set_seed

# # CONST
# set_seed(42)
# TRAINED_CASUAL_MODEL = "LunaticMaestro/gpt2-book-summary-generator"


# generator_model = pipeline('text-generation', model=TRAINED_CASUAL_MODEL)



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
from z_utils import get_dataframe

books_df = get_dataframe(CLEAN_DF_UNIQUE_TITLES)


def get_recommendation(book_title: str) -> dict:
    global generator_model
    # return "Hello"
    # # Generate hypothetical summary
    # value = generator_model("hello", max_length=50)
    fake_summaries = generate_summaries(book_title=book_title, n_samples=5) # other parameters are set to default in the function
    
    # Compute Simialrity 
    similarity, ranks = computes_similarity_w_hypothetical(hypothetical_summaries=fake_summaries)
    
    # Get ranked Documents 
    df_ranked =  books_df.iloc[ranks]
    df_ranked = df_ranked.reset_index()
    
    books = df_ranked["book_name"].to_list()[:N_RECOMMENDS]
    summaries = df_ranked["summaries"].to_list()[:N_RECOMMENDS]
    scores = similarity[ranks][:N_RECOMMENDS]

    # label wise similarity 
    label_similarity: dict = {book: score for book, score in zip(books, scores)}
    #
    # book_summaries: list[str] = [f"**{book}** \n {summary}" for book, summary in zip(books, summaries)]

    # return response
    # Generate card-style HTML
    html = "<div style='display: flex; flex-wrap: wrap; gap: 1rem;'>"
    for book, summary in zip(books, summaries):
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 1rem; width: 200px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='margin: 0;'>{book}</h3>
            <p style='font-size: 0.9rem; color: #555;'>{summaries}</p>
        </div>
        """
    html += "</div>"

    # Club the output to be processed by gradio
    response = [label_similarity, ] #html]

    return response

    return fake_summaries[0]
    # return str(value)

# We instantiate the Textbox class
textbox = gr.Textbox(label="Write random title", placeholder="The Man who knew", lines=2)
# label = gr.Label(label="Result", num_top_classes=N_RECOMMENDS)
# output = [gr.Label(label="Result", num_top_classes=N_RECOMMENDS)] + [gr.Textbox(label="Recommendation") for i in range(N_RECOMMENDS)]
output = [gr.Label(label="Similarity"), ] # gr.HTML(label="Books Descriptions")]
demo = gr.Interface(fn=get_recommendation, inputs=textbox, outputs=output)

demo.launch()
