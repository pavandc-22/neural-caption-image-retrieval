import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

class BaselineLinearRegression:
    def __init__(self, alpha=1.0):
        # Ridge is a robust version of linear regression with L2 regularization
        # It prevents overfitting when dealing with high-dimensional VGG/GloVe data
        self.model = Ridge(alpha=alpha)

    def train(self, text_features, image_features):
        """
        text_features: shape (N, D_text) - Averaged GloVe vectors for captions
        image_features: shape (N, D_img) - VGG-19 fc7 features for images
        """
        print("Training baseline linear regression model...")
        self.model.fit(text_features, image_features)
        print("Training complete.")

    def predict(self, text_features):
        """
        Maps the text vectors into the predicted image feature space.
        """
        return self.model.predict(text_features)

    def retrieve_top_k(self, query_text_feature, all_image_features, k=10):
        """
        Finds the top K most relevant images for a given text query.
        """
        # Predict the ideal image vector for the given text
        predicted_img_vector = self.predict(query_text_feature)
        
        # Calculate cosine similarity between the predicted vector and all actual image vectors
        similarities = cosine_similarity(predicted_img_vector, all_image_features)
        
        # Sort to find the indices of the top K highest similarity scores
        top_k_indices = np.argsort(similarities[0])[-k:][::-1]
        return top_k_indices

# Example usage:
# baseline = BaselineLinearRegression()
# baseline.train(train_text_features, train_image_features)
# top_10 = baseline.retrieve_top_k(test_query, all_test_images, k=10)
