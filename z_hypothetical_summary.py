# This Code is about generating supercial book summaries based on the title
#  the idea is that it will share the similar semantic meaning to the books summaries
#  already stored because the generator model is fine-tune on the book summaries.
from typing import Optional
from transformers import pipeline, set_seed

# CONST
set_seed(42)
TRAINED_CASUAL_MODEL = "LunaticMaestro/gpt2-book-summary-generator"

import gradio as gr

if gr.NO_RELOAD:
	generator_model = pipeline('text-generation', model=TRAINED_CASUAL_MODEL)


def generate_summaries(book_title: str, genre: Optional[str] = None, n_samples=2, top_k = 50, top_p = 0.85, ) -> list[str]:
    '''Generate hypothetical summaries based on book title

    Args:
        book_title: a 2-3 words descriptive name of the book
        n_samples: (default=2) count of hypothetical summaries
        top_k: (default = 50) 
        top_p: (default=0.85)
    Returns: 
        summaries: list of hypothetical summaries.
    '''
    global generator_model

    # basic prompt very similary to one used in fine-tuning
    prompt = f'''Book Title: {book_title}
Description: {book_title}'''

    # Use genre if provied, in the same order as while training
    if genre: 
        prompt = "Genre: {genre}]\n"+prompt
    
    # summaries generation
    output: list[dict] = generator_model(prompt, max_length=100, num_return_sequences=n_samples, top_k=top_k, top_p=top_p)

    # post-processing summaries to remove prompt which was provided
    summaries = list() 
    for summary in output: 
        summaries.append(
            summary['generated_text'].lstrip(prompt) 
        )

    return summaries