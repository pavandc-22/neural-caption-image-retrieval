import torch
import torch.nn as nn

class CaptionRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, output_size=4096, num_layers=1):
        super(CaptionRNN, self).__init__()
        # The embedding layer will be loaded with pre-trained GloVe vectors (usually 300-dimensional)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU network to process the sequence of word embeddings
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to map the RNN's hidden state to the 4096-dimensional VGG image space
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, captions, lengths):

        # Convert word indices to dense vectors
        embeddings = self.embedding(captions)
        
        # Pack the sequence to ignore padding tokens during GRU processing
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # Process through GRU
        _, hidden = self.gru(packed)
        
        # Take the final hidden state of the top GRU layer
        final_hidden = hidden[-1]
        
        # Project the text features into the 4096-dimensional visual space
        text_features = self.linear(final_hidden)
        
        return text_features

# Example usage:
# rnn_encoder = CaptionRNN(vocab_size=len(word_to_idx))
# text_features = rnn_encoder(caption_tensors, caption_lengths)