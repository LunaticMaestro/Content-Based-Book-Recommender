from z_utils import get_dataframe
import gradio as gr
from z_hypothetical_summary import generate_summaries

# CONST
CLEAN_DF_UNIQUE_TITLES = "unique_titles_books_summary.csv"
N_RECOMMENDS = 5

from transformers import pipeline, set_seed

# # CONST
# set_seed(42)
TRAINED_CASUAL_MODEL = "LunaticMaestro/gpt2-book-summary-generator"


if gr.NO_RELOAD:
    # Load store books
    books_df = get_dataframe(CLEAN_DF_UNIQUE_TITLES)

    generator_model = pipeline('text-generation', model=TRAINED_CASUAL_MODEL)

# if gr.NO_RELOAD:
#     from z_similarity import computes_similarity_w_hypothetical
#     


def get_recommendation(book_title: str) -> str:
    global generator_model
    # output = generator_model("Love")
    fake_summaries = generate_summaries(book_title=book_title, n_samples=5, model=generator_model) # other parameters are set to default in the function
    
    return fake_summaries[0]
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

    # Generate card-style HTML
    html = "<div style='display: flex; flex-wrap: wrap; gap: 1rem;'>"
    for book, summary in zip(books, summaries):
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 1rem; width: 200px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='margin: 0;'>{book}</h3>
            <p style='font-size: 0.9rem; color: #555;'>{summary}</p>
        </div>
        """
    html += "</div>"

    # Club the output to be processed by gradio
    response = [label_similarity, html]

    return response

# We instantiate the Textbox class
textbox = gr.Textbox(label="Write random title", placeholder="The Man who knew", lines=2)
# output = [gr.Label(label="Similarity"), gr.HTML(label="Books Descriptions")]
output = gr.Textbox(label="something")
demo = gr.Interface(fn=get_recommendation, inputs=textbox, outputs=output)

demo.launch()
