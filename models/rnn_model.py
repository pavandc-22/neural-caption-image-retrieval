import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """f_i(i) = W_i . f_VGG(i)  (trainable linear map 4096 -> shared space)."""

    def __init__(self, in_dim=4096, embed_dim=1024):
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)


class CaptionRNN(nn.Module):
    """f_c(c): word embeddings (GloVe-initialised, fine-tuned) -> GRU/LSTM -> last hidden -> linear."""

    def __init__(self, vocab_size, embed_size=300, hidden_size=1024, embed_dim=1024,
                 rnn_type="gru", num_layers=1, glove_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        if glove_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(glove_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, captions, lengths):
        emb = self.embedding(captions)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.rnn_type == "lstm":
            h = h[0]
        return self.fc(h[-1])


def embed(x, loss_type):
    """Final embedding transform: abs() for order embeddings (non-negative), L2-norm otherwise."""
    return x.abs() if loss_type == "order" else F.normalize(x, dim=-1)
