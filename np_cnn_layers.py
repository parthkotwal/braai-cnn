import cupy as np
import numpy
from cupyx.scipy.signal import correlate2d, convolve2d
from cupy.lib.stride_tricks import sliding_window_view


# Layer interface
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, d_out, lr):
        pass

    def get_parameters(self):
        params = []
        if hasattr(self, 'kernels'):
            params.append(self.kernels)
        if hasattr(self, 'biases'):
            params.append(self.biases)
        if hasattr(self, 'weights'):
            params.append(self.weights)
        return params

    def get_gradients(self):
        grads = []
        if hasattr(self, 'kernels_grad'):
            grads.append(self.kernels_grad)
        if hasattr(self, 'biases_grad'):
            grads.append(self.biases_grad)
        if hasattr(self, 'weights_grad'):
            grads.append(self.weights_grad)
        return grads


    def set_parameters(self, params):
        i = 0
        if hasattr(self, 'kernels'):
            self.kernels = params[i]
            i += 1
        if hasattr(self, 'biases'):
            self.biases = params[i]
            i += 1
        if hasattr(self, 'weights'):
            self.weights = params[i]
            i += 1

class Adam:
    def __init__(self, params, lr, beta1, beta2, epsilon):
        self.params = params
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i in range(len(self.params)):
            g = grads[i]
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1)*g
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2)*(g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            self.params[i] -= self.lr * (m_hat / (self.epsilon + np.sqrt(v_hat)))

    def collect_params(self):
        return self.params

# Converts NCHW input to column matrix
def im2col(input, KH, KW, stride=1):
      N, C, H, W = input.shape
      OH = (H - KH) // stride + 1
      OW = (W - KW) // stride + 1

      patches = sliding_window_view(input, (KH, KW), axis=(2,3))
      patches = patches[:, :, ::stride, ::stride, :, :]
      cols = patches.transpose(0,2,3,1,4,5).reshape(N*OH*OW, -1)
      return cols, OH, OW

# Reverse of im2col for accumulating gradients to x
def col2im(cols, input_shape, KH, KW, stride=1):
    N, C, H, W = input_shape
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1

    patches = cols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    x_grad = np.zeros(input_shape, dtype=cols.dtype)

    for i in range(KH):
        for j in range(KW):
            x_grad[:, :, i:i+stride*OH:stride, j:j+stride*OW:stride] += patches[:, :, i, j]
    return x_grad

class Conv2D(Layer):
    def __init__(self, kernel_size, output_depth):
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.initialized = False

    def initialize_layer(self, input_shape):
        N, self.input_depth, H, W = input_shape
        self.KH = self.kernel_size
        self.KW = self.kernel_size
        self.OH = H - self.KH + 1
        self.OW = W - self.KW + 1

        fan_in = self.input_depth * self.KH * self.KW
        self.kernels = np.random.randn(self.output_depth, self.input_depth, self.KH, self.KW) * np.sqrt(2.0 / fan_in)
        self.biases = np.random.randn(self.output_depth)
        self.initialized = True

    def forward(self, input):
        if not self.initialized:
            self.initialize_layer(input.shape)

        self.input = input
        # 1. im2 col: all patches in one big matrix
        cols = im2col(input, self.KH, self.KW)[0]

        # 2. reshape kernels to 2D
        W_col = self.kernels.reshape(self.output_depth, -1)

        # 3. GEMM
        out_cols = cols @ W_col.T
        out_cols += self.biases

        # 4. Reshape back to OG
        N = input.shape[0]
        self.output = out_cols.reshape(N, self.OH, self.OW, self.output_depth).transpose(0,3,1,2)
        return self.output

    def backward(self, d_out, lr):
        N = d_out.shape[0]

        # 1. flatten d_out to columns
        d_cols = d_out.transpose(0,2,3,1).reshape(-1, self.output_depth)

        # 2. grad kernels
        cols = im2col(self.input, self.KH, self.KW)[0]
        W_col_grad = d_cols.T @ cols
        self.kernels_grad = W_col_grad.reshape(self.kernels.shape) / N

        # 3. grad input cols
        W_col = self.kernels.reshape(self.output_depth, -1)
        cols_grad = d_cols @ W_col

        # 4. col2im to get input gradient
        input_grad = col2im(cols_grad, self.input.shape, self.KH, self.KW)

        # 5) bias grad
        self.biases_grad = d_out.sum(axis=(0,2,3)) / N
        return input_grad

    def get_gradients(self):
      return [self.kernels_grad, self.biases_grad]
    
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, d_out, lr):
        # Relu prime
        # print(f"{self.__class__.__name__} received grad of shape {d_out.shape}")

        return d_out * (self.input > 0)
    
class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape # (N, C, H, W)
        return input.reshape(self.input_shape[0], -1) # (N, C*H*W)

    def backward(self, d_out, lr=None):
        # print(f"{self.__class__.__name__} received grad of shape {d_out.shape}")

        return d_out.reshape(self.input_shape)
    
class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        N, C, H, W = input.shape

        patches, OH, OW = im2col(input, self.pool_size, self.pool_size, stride=self.stride)
        patches = patches.reshape(-1, C, self.pool_size*self.pool_size)
        self.argmax = patches.argmax(axis=2)

        out = patches.max(axis=2)
        out = out.reshape(N, OH, OW, C).transpose(0,3,1,2)
        return out

    def backward(self, d_out, lr=None):
        N, C, H, W = self.input.shape
        # print(d_out.shape)
        dum1, dum2, OH, OW = d_out.shape

        d_flat = d_out.transpose(0,2,3,1).reshape(-1, C)
        patches_grad = np.zeros((d_flat.shape[0], C, self.pool_size*self.pool_size), dtype=d_flat.dtype)

        i = np.arange(patches_grad.shape[0])[:,None]
        patches_grad[i, np.arange(C), self.argmax] = d_flat

        cols_grad = patches_grad.reshape(-1, C*self.pool_size*self.pool_size)
        return col2im(cols_grad, self.input.shape, self.pool_size, self.pool_size, stride=self.stride)
    
class FC(Layer):
    def __init__(self, input_size=None, output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.biases = None

    def forward(self, input):
        self.input = input

        if self.weights is None and self.output_size is not None:
            self.input_size = input.shape[-1]
            # self.weights = np.random.randn(self.output_size, self.input_size)
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = np.random.uniform(-limit, limit, (self.output_size, self.input_size))
            self.biases = np.zeros(self.output_size)

        if input.ndim == 1:
            # print("Final FC output before sigmoid:", input[:5].flatten())
            return np.dot(self.weights, input) + self.biases
        else:
            # print("Final FC output before sigmoid:", input[:5].flatten())

            return np.dot(input, self.weights.T) + self.biases

    def backward(self, d_out, lr):
        # print(f"[FC backward] d_out.shape={d_out.shape}, input.shape={self.input.shape}, weights.shape={self.weights.shape}")
        if self.input.ndim == 1 and self.weights is not None:
            weights_gradient = np.outer(d_out, self.input)
            input_gradient = np.dot(self.weights.T, d_out)
        else:
            weights_gradient = np.dot(d_out.T, self.input) / d_out.shape[0]
            input_gradient = np.dot(d_out, self.weights)

        self.weights_grad = weights_gradient
        self.biases_grad = d_out.mean(axis=0) if d_out.ndim > 1 else d_out

        return input_gradient

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, input):
        if self.training:
            self.mask = (np.random.rand(*input.shape) > self.rate).astype(input.dtype)
            return input * self.mask / (1.0 - self.rate)
        else:
            return input

    def backward(self, d_out, lr=None):
        if self.training:
            return d_out * self.mask / (1.0 - self.rate)
        else:
            return d_out

class Sigmoid(Layer):
    def forward(self, input):
        self.input = np.clip(input, -500, 500)
        self.out = 1 / (1 + np.exp(-self.input))
        return self.out

    def backward(self, d_out, lr=None):
        # print(f"{self.__class__.__name__} received grad of shape {d_out.shape}")
        return d_out * self.out * (1-self.out)
    
def BCE(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-(1e-15))
    return -(y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def BCEG(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-(1e-15))
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

class NumpyCNN:
    def __init__(self):
        self.layers = [
            # Block 1
            Conv2D(kernel_size=3, output_depth=16), ReLU(),
            Conv2D(kernel_size=3, output_depth=16), ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Dropout(dropout_rate=0.25),

            # Block 2
            Conv2D(kernel_size=3, output_depth=32), ReLU(),
            Conv2D(kernel_size=3, output_depth=32), ReLU(),
            MaxPool2D(pool_size=2, stride=4),
            Dropout(dropout_rate=0.25),

            # Fully connected
            Flatten(),
            FC(input_size=1152, output_size=256), ReLU(),
            Dropout(dropout_rate=0.5),
            FC(input_size=256, output_size=1), Sigmoid()
        ]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                layer.training = True

            # print(f"Before layer #{i+1} {layer.__class__.__name__}: Input shape is {x.shape}")
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad

    def predict(self, x):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = False
            x = layer.forward(x)
        return x

    def save(self, filename):
        save_dict = {}
        for i, layer in enumerate(self.layers):
            ps = layer.get_parameters()
            cpu_ps = [np.asnumpy(p) for p in ps]
            for j, p in enumerate(cpu_ps):
                save_dict[f"layer_{i}_param_{j}"] = p
        numpy.savez(filename, **save_dict)
        print(f"Saved model at {filename}")


    def load(self, filename, input_shape):
        dummy_input = np.zeros(input_shape)
        self.predict(dummy_input)

        data = numpy.load(filename, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            ps = []
            j = 0
            while True:
                key = f"layer_{i}_param_{j}"
                if key in data:
                    ps.append(np.array(data[key]))
                    j += 1
                else:
                    break
            if ps:
                layer.set_parameters(ps)


        print(f"Model from {filename} loaded into {type(self).__name__}")


    def collect_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params

    def collect_gradients(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "get_gradients"):
                grads.extend(layer.get_gradients())
        return grads
    
