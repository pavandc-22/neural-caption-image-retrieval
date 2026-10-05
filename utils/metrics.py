import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def calculate_recall_at_k(similarity_matrix, k_values=(1, 5, 10)):
    """
    Calculates Recall@K metrics from a query-to-image similarity matrix.
    
    similarity_matrix: array of shape (N_queries, N_images)
                      where row i contains similarity scores for query i against all images.
                      Assumes query i corresponds to image i.
    k_values: Tuple of K thresholds to evaluate.
    """
    num_queries = similarity_matrix.shape[0]
    recalls = {}
    
    # Sort indices in descending order of similarity score
    ranked_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    
    for k in k_values:
        top_k = ranked_indices[:, :k]
        
        # Check if ground truth index i exists within top k predictions for query i
        correct_retrievals = sum(1 for i in range(num_queries) if i in top_k[i])
        recalls[f"R@{k}"] = (correct_retrievals / num_queries) * 100.0
        
    return recalls

def evaluate_retrieval(text_features, image_features, k_values=(1, 5, 10)):
    """
    Computes pairwise cosine similarities and returns Recall@K scores.
    
    text_features: NumPy array or PyTorch tensor of shape (N, D)
    image_features: NumPy array or PyTorch tensor of shape (N, D)
    """
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.detach().cpu().numpy()
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.detach().cpu().numpy()
        
    # Compute full pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(text_features, image_features)
    
    return calculate_recall_at_k(sim_matrix, k_values=k_values)

# Example usage:
# scores = evaluate_retrieval(predicted_text_vectors, target_image_vectors)
# print(scores)  # e.g., {'R@1': 18.5, 'R@5': 45.2, 'R@10': 62.1}