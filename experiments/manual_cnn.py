"""NumPy CNN forward/backward from the user's A04, extracted without notebook execution."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
SEED=42

def take_limit(X, y, limit):
    if limit is None or limit >= len(X):
        return (X, y)
    return (X[:limit], y[:limit])

class Conv2DSameManual:

    def __init__(self, in_channels, out_channels, kernel_size=3, seed=42):
        rng = np.random.default_rng(seed)
        self.k = kernel_size
        self.pad = kernel_size // 2
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = (rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * scale).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x):
        self.x = x
        p = self.pad
        self.xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        windows = sliding_window_view(self.xp, (self.k, self.k), axis=(2, 3))
        return np.einsum('nchwkl,fckl->nfhw', windows, self.W) + self.b[None, :, None, None]

    def backward(self, dout, lr):
        windows = sliding_window_view(self.xp, (self.k, self.k), axis=(2, 3))
        dW = np.einsum('nfhw,nchwkl->fckl', dout, windows)
        db = dout.sum(axis=(0, 2, 3))
        dxp = np.zeros_like(self.xp)
        H, W = (dout.shape[2], dout.shape[3])
        for i in range(self.k):
            for j in range(self.k):
                dxp[:, :, i:i + H, j:j + W] += np.einsum('nfhw,fc->nchw', dout, self.W[:, :, i, j])
        p = self.pad
        dx = dxp[:, :, p:-p, p:-p] if p > 0 else dxp
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class DenseManual:

    def __init__(self, in_dim, out_dim, seed=42):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dx = dout @ self.W.T
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        self.W -= lr * dW
        self.b -= lr * db
        return dx

def relu(x):
    return np.maximum(x, 0)

def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    p = expz / expz.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1
    d /= n
    return (float(loss), d)

class MaxPool2x2Manual:

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        H2, W2 = (H // 2, W // 2)
        x_crop = x[:, :, :H2 * 2, :W2 * 2]
        xr = x_crop.reshape(N, C, H2, 2, W2, 2)
        self.maxv = xr.max(axis=(3, 5), keepdims=True)
        self.crop_shape = x_crop.shape
        return self.maxv.squeeze((3, 5))

    def backward(self, dout):
        x = self.x
        N, C, H, W = x.shape
        H2, W2 = (H // 2, W // 2)
        x_crop = x[:, :, :H2 * 2, :W2 * 2]
        xr = x_crop.reshape(N, C, H2, 2, W2, 2)
        mask = xr == self.maxv
        denom = mask.sum(axis=(3, 5), keepdims=True)
        dxr = mask * (dout[:, :, :, None, :, None] / denom)
        dx = np.zeros_like(x)
        dx[:, :, :H2 * 2, :W2 * 2] = dxr.reshape(self.crop_shape)
        return dx

class ManualMNISTCNN:

    def __init__(self, seed=42):
        self.conv1 = Conv2DSameManual(1, 4, 3, seed)
        self.conv2 = Conv2DSameManual(4, 8, 3, seed + 1)
        self.pool = MaxPool2x2Manual()
        self.fc = DenseManual(8 * 14 * 14, 10, seed + 2)

    def forward(self, x):
        self.z1 = self.conv1.forward(x)
        self.h1 = relu(self.z1)
        self.z2 = self.conv2.forward(self.h1)
        self.h2 = relu(self.z2)
        self.p = self.pool.forward(self.h2)
        self.pool_shape = self.p.shape
        return self.fc.forward(self.p.reshape(len(x), -1))

    def train_batch(self, x, y, lr):
        logits = self.forward(x)
        loss, dlogits = softmax_ce(logits, y)
        dp = self.fc.backward(dlogits, lr).reshape(self.pool_shape)
        dh2 = self.pool.backward(dp)
        dz2 = dh2 * (self.z2 > 0)
        dh1 = self.conv2.backward(dz2, lr)
        dz1 = dh1 * (self.z1 > 0)
        self.conv1.backward(dz1, lr)
        return loss

    def predict(self, X, batch_size=256):
        out = []
        for s in range(0, len(X), batch_size):
            out.append(np.argmax(self.forward(X[s:s + batch_size]), axis=1))
        return np.concatenate(out)

def train_manual(model, X, y, epochs=3, batch_size=64, lr=0.005):
    rng = np.random.default_rng(SEED)
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(X))
        losses = []
        for s in range(0, len(X), batch_size):
            b = order[s:s + batch_size]
            losses.append(model.train_batch(X[b], y[b], lr))
        hist.append(float(np.mean(losses)))
        print(f'Epoch {ep + 1}/{epochs} - loss={hist[-1]:.4f}')
    return hist
