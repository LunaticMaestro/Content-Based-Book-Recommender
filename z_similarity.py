from z_utils import load_cache_embeddings 
from z_embedding import model, get_embeddings
import torch 
import numpy as np

books_summaries_embs = load_cache_embeddings()

def computes_similarity_w_hypothetical(hypothetical_summaries: list[str]) -> (np.ndarray, np.ndarray):
    '''Computes cosine similarity between stored book_summaries and all hypothetical_summaries
    
    Returns: 
        
        Avg cosine similiarity between actual books sumamries' embeddings and hypothetical summaries
        
        Ranks of the books summaries based on above consine similarity Distance; Lower ranks means more similar
    '''
    global books_summaries_embs, model
    hypothetical_summaries_embs = get_embeddings(hypothetical_summaries)
    similarity: torch.Tensor = model.similarity(books_summaries_embs, hypothetical_summaries_embs)

    # Average ouut the distance across all hypothetical embddings
    similarity = torch.mean(similarity, dim=1)

    # Get the order
    ranks = torch.argsort(similarity, descending=True)

    # return None
    return similarity.detach().numpy(), ranks.detach().numpy()


