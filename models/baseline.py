import numpy as np


class BaselineLinearRegression:
    """Multi-response ridge regression  W = argmin sum_k ||f_VGG(i_k) - W f_GloVe(c_k)||^2 + alpha||W||^2
    from averaged-GloVe caption vectors to VGG fc7 image vectors (closed form).
    Retrieval = nearest image (Euclidean) to the predicted image vector.
    Captions are image-major: caption j belongs to image j // caps_per_img (avoids materialising 5x targets)."""

    def __init__(self, alpha=1.0, caps_per_img=5):
        self.alpha, self.cpi = alpha, caps_per_img

    def fit(self, text_feats, img_feats):
        X = text_feats.astype(np.float64)
        Y = img_feats.astype(np.float64)
        self.mx, self.my = X.mean(0), Y.mean(0)           # each image has the same #captions -> same mean
        Xc, Yc = X - self.mx, Y - self.my
        group = Xc.reshape(len(Y), self.cpi, -1).sum(1)   # (N, D): sum of caption vectors per image
        xtx = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(xtx, group.T @ Yc)       # (D_text, 4096)
        return self

    def predict(self, text_feats):
        return (text_feats - self.mx) @ self.W + self.my

    def similarity(self, text_feats, gallery_img_feats):
        """(Q,N) similarity = -squared Euclidean distance between predicted and real image features."""
        p = self.predict(text_feats).astype(np.float32)
        g = gallery_img_feats.astype(np.float32)
        return -((p ** 2).sum(1, keepdims=True) - 2 * p @ g.T + (g ** 2).sum(1)[None])
