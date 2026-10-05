import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rnn_model import CaptionRNN
from models.extractors import VGG19Extractor
from utils.data_loader import CocoImageCaptionDataset
from utils.metrics import evaluate_retrieval

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, text_embeds, img_embeds):
        """
        Calculates pairwise margin-based ranking loss for positive and negative pairs.
        text_embeds: (Batch_Size, 4096)
        img_embeds: (Batch_Size, 4096)
        """
        # Cosine similarity matrix between all captions and all images in the batch
        text_norm = nn.functional.normalize(text_embeds, p=2, dim=1)
        img_norm = nn.functional.normalize(img_embeds, p=2, dim=1)
        scores = torch.mm(text_norm, img_norm.t()) # (Batch_Size, Batch_Size)
        
        diagonal = scores.diag().view(-1, 1) # Positive pair similarity scores
        
        # Calculate margins for mismatched text-image pairs
        cost_s = torch.clamp(self.margin - diagonal + scores, min=0) # Image contrastive loss
        cost_im = torch.clamp(self.margin - diagonal.t() + scores, min=0) # Text contrastive loss
        
        # Zero out self-contrastive diagonal terms
        mask = torch.eye(scores.size(0)).bool().to(scores.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)
        
        return cost_s.sum() + cost_im.sum()

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = 10000  # Adjust based on your GloVe dictionary size
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    margin = 0.2

    # Instantiate models
    text_encoder = CaptionRNN(vocab_size=vocab_size).to(device)
    vgg_extractor = VGG19Extractor().to(device)
    vgg_extractor.eval() # VGG features remain frozen

    criterion = MarginRankingLoss(margin=margin)
    optimizer = optim.Adam(text_encoder.parameters(), lr=learning_rate)

    print("Starting training loop...")
    for epoch in range(num_epochs):
        text_encoder.train()
        total_loss = 0.0

        # Dummy batch simulation loop (Replace with actual DataLoader iterator)
        for step in range(100):
            # Simulated tensors for execution validation:
            # Replace dummy_captions with dataset outputs in production
            dummy_captions = torch.randint(0, vocab_size, (batch_size, 15)).to(device)
            dummy_lengths = torch.full((batch_size,), 15, dtype=torch.int64)
            dummy_images = torch.randn(batch_size, 10, 3, 224, 224).to(device)

            optimizer.zero_grad()

            # Extract image and text features
            with torch.no_grad():
                img_features = vgg_extractor(dummy_images)
            text_features = text_encoder(dummy_captions, dummy_lengths)

            # Compute loss
            loss = criterion(text_features, img_features)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # Save model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(text_encoder.state_dict(), "checkpoints/gru_caption_model.pth")
    print("Model saved to checkpoints/gru_caption_model.pth")

if __name__ == "__main__":
    train_model()
