# `braai-cnn`: Real/Bogus Transient Detection using CNNs and Classical ML

This is a deep learning project focused on distinguishing true astrophysical events from false detections in Zwicky Transient Facility (ZTF) imaging data. The project reimplements Duev et al. (2019)'s Bogus/Real Adversarial AI (`braai`) CNN in both `PyTorch` and `CuPy` (GPU-accelerated `NumPy`) and evaluates them against classical machine learning models such as Logistic Regression and Random Forest classifiers. 

**Note: While this project is inspired by the BRAAI paper's approach to real/bogus classification in ZTF data, it does not attempt a strict reproduction of their results. Due to differences in available data splits, batch sizes, and RB score thresholding, my model results are not directly comparable to those in the original study. However, the architectural inspiration of the model and core problem we aim to address remain aligned.** ❤️

## Problem
Time domain analysis has been central to astronomical research since the Scientific Revolution in the 1600s. 
In an effort to identify transient objects that rapidly change in brightness or position (comets, gamma ray bursts, supernovae, etc.), modern real-time detection systems like the Zwicky Transient Facility (ZTF) generate massive volumes of candidate events per night. However, many of these events are spurious (bogus) detections caused by instrumental noise, data processing errors, or other artifacts (false alarms). ZTF alone generates hundreds of thousands of candidates nightly, thus making manual inspection infeasible and demonstrating a need for automated, but highly accurate, classification systems. The challenge lies in correctly differentiating true transients from bogus detections, often using only a small image cutout around each candidate. These cutouts are often noisy, low-resolution, and can contain subtle features that even experts may disagree on.

This is known as the **real/bogus classification** task, and it is foundational in automating modern sky surveys. Machine learning algorithms, especially deep learning models, are frequently employed for their ability to pick up on patterns in these cutouts and automate their way through billions of candidates. One such example is the `braai` CNN, a deep model proposed by Duev et al. (2019), which presented expectional performance on ZTF imaging data using convolutional architectures. In contrast to earlier models based on flattened pixel arrays or hand-engineered features, their work underlined spatial structure in this specific classification task.

## Dataset
- **Source**: [University of Washington's ZTF Alert Archive](https://ztf.uw.edu/alerts/public/)
    - March 2nd, 2025 dataset
- **Size**: 38,368 labeled samples
    - 3 Channels: Science, Template, Difference
    - Padded to 63 x 63 
    - Normalized to `[0,1]`
- **Labels**:
    - 0 = Bogus (~20,274)
    - 1 = Real (~18,094)
        - Threshold: RB Score > 0.7 in raw data 
- **Split**: 
    - Train Size: 75% (~28,776)
    - Test Size: 25% (~9,592)

![Bogus Image Example](/images/bogus_example.png)

![Real Image Example](/images/real_example.png)
## Models

### `braai` CNN

![braai architecture](/images/fig-braai.png)

`braai` operates directly on 3-channel image cutout triplets (science, template, difference). Each image is normalized and stacked to form a `63 x 63 x 6` input tensor. Its design emphasizes the preservation of spatial structure in low-resolution astronomical images. The architecture VGG6 is a simplified version of VGG18, a CNN developed by the Visual Geometry Group (VGG) at the University of Oxford. It consists of:
- 4 convolutional layers:
    - First two: 16 filters of size `3 x 3`
    - Second two: 32 filters of size `3 x 3`
- 2 fully connected (FC) layers
- 3 dropout layers for regularization:
    - 0.25 for each max-pooling layer
    - 0.5 after the second FC layer
- Activation Functions:
    - ReLU for all trainable layers
    - Sigmoid on the output layer to return real/bogus score in `[0,1]`
- Binary Cross-Entropy Loss Function
- Adam Optimizer

With a dataset size of ~30,000 images, the data was split into 81% train, 9% validation, and 10% test, and used to train in batch sizes of 64 samples. Additionally, the model trained for about ~150-200 epochs with early stopping, which lasted 20 minutes, while utilizing an NVIDIA Tesla P100 GPU (12 GB).

`braai` served as an architectural and conceptual reference for the models in my project, however differed in training data, data splits, key hyperparameters (ex: learning rate), and frameworks.

### 1. `CuPy`/`NumPy` CNN
This model was trained with an NVIDIA L4 Tensor Core GPU and had a training time of ~35 minutes. As mentioned before, this CNN resembles `braai` in terms of several characteristics such as architecture (see above), optimizer, input shape and more. 

Each layer (`Conv2D`, `ReLU`, `FC`, etc.) is implemented as a modular class with its own `.forward()` and `.backward()` methods. The `Dropout` layer is toggled per mode (train/eval). The `Conv2D` layers were initialized with He initialization, a common weight initialization method for ReLu activated layers. However, the `FC` layer used Xavier due to exploding gradients that were seen in initial training runs. 

The training loop was equipped with early stopping using validation loss (patience = 5). Hyperparameters included a batch size = `64`, learning rate = `0.001`, and validation split = `0.2`. Training stopped at 20 epochs.

The following snippet shows the layer-by-layer forward and backward pass functions over the network.

```
class NumpyCNN:
    # Other functions
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                layer.training = True
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad
```

After `.forward()`, the mean of the binary cross entropy loss for each predicted value is calculated and used to update the total loss. The `BCEG()` function computes the derivative/gradient of the BCE loss with respect to `y_pred` and passed backwards through the network (only during training).

```
def BCE(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-(1e-15))
    return -(y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def BCEG(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-(1e-15))
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
```

To accelerate computation on the GPU (significantly), the `Conv2D` and `MaxPool2D` layers internally use **im2col** and **col2im** techniques. These convert lengthy spatial operations into more efficient matrix multiplications that GPUs can compute in parallel. This critical change from direct cross-correlation and convolution was implemented after finding that training would have originally lasted ~33 days.

```
from cupy.lib.stride_tricks import sliding_window_view

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
```
Here is a GIF/JIF that illustrates this genius technique:

![im2col GIF](/images/im2col.gif)



## Evaluation

## Key Outcomes

## Learnings
