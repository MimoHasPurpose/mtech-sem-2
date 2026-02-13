import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr
        
        # Xavier initialization
        self.Wx = np.random.randn(hidden_size, input_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """Forward pass through sequence"""
        h_prev = np.zeros((self.hidden_size, 1))
        self.cache = []

        outputs = []
        for x in inputs:
            x = x.reshape(-1, 1)
            
            h = np.tanh(self.Wx @ x + self.Wh @ h_prev + self.bh)
            y = self.Wy @ h + self.by
            
            self.cache.append((x, h_prev, h))
            h_prev = h
            outputs.append(y)

        return outputs

    def backward(self, dy_list):
        """Backpropagation Through Time"""
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.cache))):
            x, h_prev, h = self.cache[t]
            dy = dy_list[t]

            dWy += dy @ h.T
            dby += dy

            dh = self.Wy.T @ dy + dh_next
            dh_raw = (1 - h**2) * dh

            dWx += dh_raw @ x.T
            dWh += dh_raw @ h_prev.T
            dbh += dh_raw

            dh_next = self.Wh.T @ dh_raw

        # Gradient descent update
        for param, dparam in zip(
            [self.Wx, self.Wh, self.Wy, self.bh, self.by],
            [dWx, dWh, dWy, dbh, dby]
        ):
            param -= self.lr * dparam
