from z_utils import get_dataframe
import gradio as gr
from z_hypothetical_summary import generate_summaries
from z_similarity import computes_similarity_w_hypothetical
from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer


# CONST
CLEAN_DF_UNIQUE_TITLES = "unique_titles_books_summary.csv"
N_RECOMMENDS = 5
set_seed(42)
TRAINED_CASUAL_MODEL = "LunaticMaestro/gpt2-book-summary-generator"
EMB_MODEL = "all-MiniLM-L6-v2"
GRADIO_TITLE = "Content Based Book Recommender"
GRADIO_DESCRIPTION = '''
This is a [HyDE](https://arxiv.org/abs/2212.10496) based searching mechanism that generates random summaries using your input book title and matches books which has summary similary to generated ones. The books, for search, are used from used [Kaggle Dataset: arpansri/books-summary](https://www.kaggle.com/datasets/arpansri/books-summary)

**Should take ~ 15s to 30s** for inferencing.
'''

# Caching mechanism for gradio
if gr.NO_RELOAD: # Reference: https://www.gradio.app/guides/developing-faster-with-reload-mode
    # Load store books
    books_df = get_dataframe(CLEAN_DF_UNIQUE_TITLES)

    # Load generator model
    generator_model = pipeline('text-generation', model=TRAINED_CASUAL_MODEL)

    # Load embedding model 
    emb_model = SentenceTransformer(EMB_MODEL)

def get_recommendation(book_title: str) -> list:
    '''Returns data model suitable to be render in gradio interface;

    Args:
        book_title: the book name you are looking for

    Returns 
     list of two values; firs value is a dictionary of <book, similarity_score>; Second Value is the card view in html generated form
    '''
    global generator_model, emb_model

    # output = generator_model("Love")
    fake_summaries = generate_summaries(book_title=book_title, n_samples=5, model=generator_model) # other parameters are set to default in the function
    
    # Compute Simialrity 
    similarity, ranks = computes_similarity_w_hypothetical(hypothetical_summaries=fake_summaries, model=emb_model)

    # Get ranked Documents 
    df_ranked =  books_df.iloc[ranks]
    df_ranked = df_ranked.reset_index()
    
    # post-process for gradio interface
    books = df_ranked["book_name"].to_list()[:N_RECOMMENDS]
    summaries = df_ranked["summaries"].to_list()[:N_RECOMMENDS]
    scores = similarity[ranks][:N_RECOMMENDS]
    #
    # For gr.Label interface
    label_similarity: dict = {book: score for book, score in zip(books, scores)}
    #
    # Generate card-style HTML; to render book names and their summaries
    html = "<div style='display: flex; flex-wrap: wrap; gap: 1rem;'>"
    for book, summary in zip(books, summaries):
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 1rem; width: 200px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='margin: 0;'>{book}</h3>
            <p style='font-size: 0.9rem; color: #555;'>{summary}</p>
        </div>
        """
    html += "</div>"

    # Club the output to be processed by gradio INterface
    response = [label_similarity, html]

    return response

# Input Interface Render
input_textbox = gr.Textbox(label="Search for book with name similary to", placeholder="Rich Dad Poor Dad", max_lines=1)

# Output Interface Render
output = [gr.Label(label="Similar Books"), gr.HTML(label="Books Descriptions", show_label=True)]

# Stich interace and run
demo = gr.Interface(
    fn=get_recommendation, 
    inputs=input_textbox, 
    outputs=output,
    title=GRADIO_TITLE,
    description=GRADIO_DESCRIPTION
)

demo.launch()
