import numpy as np
from abc import abstractmethod
from numpy.lib.stride_tricks import as_strided


class Layer:
    def __init__(self):
        self.optimizable = True

    @abstractmethod
    def forward(self, X: np.ndarray):
        pass

    @abstractmethod
    def backward(self, grads: np.ndarray):
        pass

    def frozen(self):
        self.optimizable = False



class Linear(Layer):
    def __init__(self, in_dim, out_dim, weight_initialize_method=np.random.normal, weight_decay=True, weight_decay_param=0):
        super().__init__()
        self.params = {
            'W':weight_initialize_method(size=(in_dim, out_dim)) * 0.1,
            'b':np.zeros((1, out_dim))
        }
        self.grads = {'W': None, 'b': None}

        self.input = None    # for backward process

        self.weight_decay = weight_decay
        self.weight_decay_param = weight_decay_param

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        """
        conduct forward process
        :param X: [batch_size, in_dim]
        :return: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grads: np.ndarray):
        """
        calculate the grads of the parameters
        :param grads: [batch_size, out_dim]
        :return: [batch_size, in_dim]
        """
        self.grads['W'] = self.input.T @ grads
        self.grads['b'] = np.sum(grads, axis = 0, keepdims=True)
        return grads @ self.params['W'].T

    def deactivate_weight_decay(self):
        self.weight_decay = False


class Conv2D(Layer):
    def __init__(self, in_channel, out_channel, kernel_size, weight_initialize_method=np.random.normal,
                 weight_decay=True, weight_decay_param=0):
        super().__init__()
        self.params = {
            'W': weight_initialize_method(size=(in_channel, out_channel, *kernel_size)) * 0.1,
            'b': np.zeros((out_channel, 1, 1))
        }
        self.grads = {'W': None, 'b': None}
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.input_shape = None
        self.cols = None
        self.weight_decay = weight_decay
        self.weight_decay_param = weight_decay_param

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input_shape = X.shape
        bs, in_ch, h, w = X.shape
        m, n = self.kernel_size

        # 使用im2col进行向量化
        self.cols = self.im2col(X)
        W_reshaped = self.params['W'].transpose(1, 0, 2, 3).reshape(self.out_channel, -1)

        # 矩阵乘法代替循环
        try:
            output = self.cols @ W_reshaped.T
        except Exception:
            print(e)
        H_out = h - m + 1
        W_out = w - n + 1
        output = output.reshape(bs, H_out, W_out, self.out_channel).transpose(0, 3, 1, 2)
        output += self.params['b'].reshape(1, -1, 1, 1)
        return output

    def backward(self, grads: np.ndarray):
        bs, out_ch, H_out, W_out = grads.shape
        m, n = self.kernel_size
        in_ch = self.in_channel
        h_in, w_in = self.input_shape[2], self.input_shape[3]

        # 计算权重梯度
        d_output = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channel)
        dW_reshaped = self.cols.T @ d_output
        dW = dW_reshaped.T.reshape(self.out_channel, in_ch, m, n).transpose(1, 0, 2, 3)
        self.grads['W'] = dW

        # 添加权重衰减
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_param * self.params['W']

        # 计算偏置梯度
        self.grads['b'] = grads.sum(axis=(0, 2, 3))[:, None, None]

        # 计算输入梯度（使用转置卷积）
        padded_grads = np.pad(grads, ((0, 0), (0, 0), (m - 1, m - 1), (n - 1, n - 1)), mode='constant')
        cols_padded = self.im2col(padded_grads)
        W_rot = np.rot90(self.params['W'], 2, axes=(2, 3)).transpose(1, 0, 2, 3).reshape(-1, in_ch)

        # 向量化输入梯度计算
        d_input_col = cols_padded @ W_rot
        d_input = d_input_col.reshape(bs, h_in, w_in, in_ch).transpose(0, 3, 1, 2)
        return d_input

    def im2col(self, X):
        bs, ch, h, w = X.shape
        m, n = self.kernel_size
        H_out = h - m + 1
        W_out = w - n + 1

        # 使用as_strided高效生成卷积窗口
        strides = X.strides
        strided_shape = (bs, ch, H_out, W_out, m, n)
        strided_strides = (strides[0], strides[1], strides[2], strides[3], strides[2], strides[3])

        strided = np.lib.stride_tricks.as_strided(
            X,
            shape=strided_shape,
            strides=strided_strides,
            writeable=False
        )

        # 重塑为列矩阵
        return strided.transpose(0, 2, 3, 1, 4, 5).reshape(bs * H_out * W_out, -1)

class MaxPooling(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        h, w = X.shape[-2:]
        assert not h % 2, 'X.shape[-2] can\'t be divided by 2'
        assert not w % 2, 'X.shape[-1] can\'t be divided by 2'
        output = np.zeros((X.shape[0], X.shape[1], h // 2, w // 2))
        for i in range(h // 2):
            for j in range(w // 2):
                output[:, :, i, j] = np.max(X[:, :, 2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)], axis=(2, 3))
        return output

    def backward(self, grads: np.ndarray):
        h, w = self.input.shape[-2:]
        m, n = grads.shape[-2:]
        result = np.zeros_like(self.input)
        assert h / 2 == m and w / 2 == n, 'Shape of grads don\'t satisfy the input'
        for i in range(m):
            for j in range(n):
                input_part = self.input[:, :, 2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)]
                for bs in range(self.input.shape[0]):
                    for c in range(self.input.shape[1]):
                        max_idx = np.argmax(input_part[bs, c])
                        result[bs, c, 2 * i + max_idx // 2, 2 * j + max_idx % 2] = grads[bs, c, i, j]
        return result

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        """
        perform ReLU function
        """
        self.input = X
        return np.where(X < 0, 0, X)

    def backward(self, grads: np.ndarray):
        """
        conduct backward process
        """
        return np.where(self.input < 0, 0, grads)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        return 1 / (1 + np.exp(-X))

    def backward(self, grads: np.ndarray):
        X = self.input
        return 1 / (2 + np.exp(X) + np.exp(-X)) * grads

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.frozen()
        self.input = None
        self.alpha = alpha

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        return np.where(X < 0, self.alpha * X, X)

    def backward(self, grads: np.ndarray):
        return np.where(self.input < 0, self.alpha * grads, grads)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        assert X.ndim >= 2, f'input of the Flatten layer must >= 2D, got {X.ndim}D'
        return X.reshape(self.input.shape[0], -1)

    def backward(self, grads: np.ndarray):
        return grads.reshape(self.input.shape)