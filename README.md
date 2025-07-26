# `braai-cnn`: Real/Bogus Transient Detection using CNNs and Classical ML

This is a deep learning project focused on distinguishing true astrophysical events from false detections in Zwicky Transient Facility (ZTF) imaging data. The project reimplements Duev et al. (2019)'s Bogus/Real Adversarial AI (`braai`) CNN in both `PyTorch` and `CuPy` (GPU-accelerated `NumPy`) and evaluates them against classical machine learning models such as Logistic Regression and Random Forest classifiers. 

**Note: While this project is inspired by the BRAAI paper's approach to real/bogus classification in ZTF data, it does not attempt a strict reproduction of their results. Due to differences in training data, hyperparameters, and other critical factors that shape performance, my model results are not directly comparable to those in the original study. However, the architectural inspiration of the model and core problem we aim to address remain aligned.** ❤️

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

> [CuPy CNN Training Notebook on Google Colab](https://colab.research.google.com/drive/1f3Jbdesz5PL7CKwdPn_G-7eoZoDwzVj_?usp=drive_link)

This model was trained with an NVIDIA L4 Tensor Core GPU and had a training time of ~35 minutes. As mentioned before, this CNN resembles `braai` in terms of several characteristics such as architecture (see above), optimizer, input shape and more. 

Each layer (`Conv2D`, `ReLU`, `FC`, etc.) is implemented as a modular class with its own `.forward()` and `.backward()` methods. The `Dropout` layer is toggled per mode (train/eval). The `Conv2D` layers were initialized with He initialization, a common weight initialization method for ReLu activated layers. However, the `FC` layer used Xavier due to exploding gradients that were seen in initial training runs. 

He initialization:

![He initialization](/images/he.png)

Xavier initialization:

![Xavier initialization](/images/xavier.png)

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

To accelerate computation on the GPU (significantly), the `Conv2D` and `MaxPool2D` layers internally use **im2col** and **col2im** techniques for forward and backward passes respectively. These convert the lengthy spatial operations into more efficient matrix multiplications that GPUs can compute in parallel. This critical change was implemented after finding that training would have originally lasted **~67 days**. Instead of directly performing the cross-correlation and convolution between each kernel and each of the samples' channels on the CPU, the operation is simplifed into one giant matrix multiplication that GPUs excel in. That's a **99.97% decrease in training time**. Here is a GIF/JIF that illustrates this genius technique:

![im2col GIF](/images/im2col.gif)

Here are my helper functions:

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

This model emphasized low level design decisions that can affect training stability, speed, convergence, and of course, *learning*. Though the `im2col` technique improved training time, it wasn't necessarily memory efficient. As a result, the training pipeline manually ran Python's garbage collector, `gc`, to manage GPU memory occupied by large batch sizes. Additionally, the use of an optimizer also determined how backpropagation here differed from traditional implementations.

### 2. `PyTorch` CNN

> [PyTorch CNN Training Notebook on Google Colab](https://colab.research.google.com/drive/1f3Jbdesz5PL7CKwdPn_G-7eoZoDwzVj_?usp=drive_link)

To benchmark my custom `CuPy` model against a high level framework, I recreated the same VGG6 architecture in `PyTorch`. Training data, optimizer, learning rate, and other parameters were the same as the `CuPy` model (see above). This network was trained using an NVIDIA T4 GPU and lasted for a bit under 3 minutes, stopping at Epoch 38. Implementation was rather straightforward, due to PyTorch's modular API and having already implemented a low-level version of the model previously.

```
import torch.nn as nn
import torch.nn.functional as F

class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()

        # Block 1:
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        # Block 2:
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(stride=4, kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # Block 1:
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2:
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x
```

### 3. Logistic Regression

> [Baseline Model Training Notebook](/images/1_baseline_model.ipynb)

To establish a classical baseline, I trained a simple Logistic Regression model on the same dataset used for the deep learning models. Each 3 x 63 × 63 cutout was flattened into a single 11,907 dimensional vector, which discarded the data's spatial structure. Shape: `(28776, 11907)`

Both the Logistic Regression and Random Forest were trained locally on a Macbook Pro M4; training lasted for ~2 and ~5 minutes respectively (see below). 

I used `scikit-learn`'s `LogisticRegression` model, with the following hyperparameters:
- `max_iter = 1000`
- `random_state = 42`
- All other hyperparameters were default

```
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(max_iter=1000, random_state=42)
lrc.fit(X_flat_train, y_train)
```

Unlike the CNNs, this model has no awareness of spatial structure. Each pixel is treated as another feature.

### 4. Random Forest

> [Baseline Model Training Notebook](/images/1_baseline_model.ipynb)

To complement the linearity of Logistic Regression, I also trained a Random Forest Classifier using the same flattened input features. The Random Forest model served as a stronger classical baseline with its ability to handle higher dimensional inputs.

```
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_flat_train, y_train)
```

## Evaluation

All models were evaluated on the same held-out test set of 9,592 labelled samples in a unified pipeline.

Key metrics included:

- Accuracy, Precision, Recall, F1 Score, ROC AUC
- FNR–FPR at various decision thresholds
- Inference time per sample
- Confusion matrices
- Visual comparisons of ROC, PRC, and threshold behavior

### Classification Metrics
Here are each of the models ranked by their metrics on the test data.

#### Accuracy

| Rank | Model Name       | Accuracy   |
|------|------------------|------------|
| 1    | PyTorch CNN      | 0.8261     |
| 2    | NumPy CNN        | 0.8017     |
| 3    | Random Forest    | 0.7750     |
| 4    | Logistic Regression | 0.6617 |

Proportion of correct predictions out of all predictions. As our dataset was only slightly imbalanced, accuracy is a great indicator of performance here. Both CNNs outperform the baseline models, however the Random Forest is not far off.

#### Precision

| Rank | Model Name       | Precision  |
|------|------------------|------------|
| 1    | PyTorch CNN      | 0.7996     |
| 2    | Random Forest    | 0.7717     |
| 3    | NumPy CNN        | 0.7586     |
| 4    | Logistic Regression | 0.6440 |

Proportion of positive predictions that were correct. Higher precision is crucial when false positives are costly (ex: triggering follow ups). Surprisingly, the NumPy CNN falls behind the Random Forest.


#### Recall

| Rank | Model Name       | Recall     |
|------|------------------|------------|
| 1    | NumPy CNN        | 0.8499     |
| 2    | PyTorch CNN      | 0.8424     |
| 3    | Random Forest    | 0.7426     |
| 4    | Logistic Regression | 0.6317 |

Proportion of positives that were predicted correctly. Recall is significant in this situation to minimize false negatives. As the following PR curve shows, the PyTorch CNN is clearly superior, maintaining a high precision even as recall increases.

![PR Curve](/images/prc.png)


#### F1 Score

| Rank | Model Name       | F1 Score   |
|------|------------------|------------|
| 1    | PyTorch CNN      | 0.8204     |
| 2    | NumPy CNN        | 0.8017     |
| 3    | Random Forest    | 0.7569     |
| 4    | Logistic Regression | 0.6378 |

Harmonic mean of precision and recall. A balanced metric when both false positives and false negatives matter.


#### ROC AUC

| Rank | Model Name       | ROC AUC    |
|------|------------------|------------|
| 1    | PyTorch CNN      | 0.9105     |
| 2    | NumPy CNN        | 0.8902     |
| 3    | Random Forest    | 0.8551     |
| 4    | Logistic Regression | 0.7113 |

Area under Receiver Operating Characteristic Curve. The area represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative one. In other words, it summarizes the model's ability to distinguish between two classes. 

![ROC Curve](/images/roc.png)


### FPR vs FNR Analysis

The following plots mirror an analysis done in the `braai` paper and show where models balance false alarms vs. missed detections

Logistic Regression:

![Logistic Regression FPR vs FNR Curve](/images/fnrfpr_logreg.png)

Random Forest:

![Random Forest FPR vs FNR Curve](/images/fnrfpr_forest.png)

`CuPy`/`NumPy` CNN:

![NumPy CNN FPR vs FNR Curve](/images/fnrfpr_numpy.png)

`PyTorch` CNN:

![PyTorch CNN FPR vs FNR Curve](/images/fnrfpr_torch.png)

### Confusion Matrices

| Model             | Confusion Matrix |
|------------------|------------------|
| Logistic Regression | ![LogReg CM](/images/matrix_logreg.png) |
| Random Forest       | ![RF CM](/images/matrix_forest.png)         |
| CuPy CNN            | ![NumPy CM](/images/matrix_numpy.png)     |
| PyTorch CNN         | ![Torch CM](/images/matrix_torch.png)   |

### Inference Times

| Rank | Model Name       | Inference Time (s) |
|------|------------------|--------------------|
| 1    | Logistic Regression | 0.2920          |
| 2    | PyTorch CNN      | 0.4212             |
| 3    | Random Forest    | 0.5734             |
| 4    | NumPy CNN        | 78.5150            |

Finally, I calculated each models' inference time, or how quickly each model can make predictions. Logistic Regression was the fastest but also had the poorest performance, while the NumPy CNN is incredibly slower.

### Summary Table
| Model Name       | Accuracy   | Precision | Recall    | F1 Score  | ROC AUC   | FNR=FPR Threshold | FNR@Threshold (%) | FPR@Threshold (%) | Inference Time (s) |
|------------------|------------|-----------|-----------|-----------|-----------|--------------------|--------------------|--------------------|---------------------|
| Logistic Regression | 0.6617     | 0.6440    | 0.6317    | 0.6378    | 0.7113    | 0.4635             | 33.47              | 33.56              | 0.2920              |
| Random Forest     | 0.7750     | 0.7717    | 0.7426    | 0.7569    | 0.8551    | 0.4805             | 22.82              | 22.37              | 0.5734              |
| NumPy CNN         | 0.8017     | 0.7586    | 0.8499    | 0.8017    | 0.8902    | 0.5546             | 19.77              | 19.67              | 78.5150             |
| PyTorch CNN       | 0.8261     | 0.7996    | 0.8424    | 0.8204    | 0.9105    | 0.5295             | 17.31              | 17.26              | 0.4212              |

## Key Outcomes

The evaluation results illustrated clear distinctions in performance, trade off, and practical aspects across all four models:

### 1. Deep learning > classical models

The `PyTorch` and `NumPy` CNNs dominated the Logistic Regression and Random Forest baseline models in nearly every category:

- `PyTorch` CNN achieved the highest accuracy (82.6%), F1 Score (82.0%), and ROC AUC (0.9105). This is the the most balanced and reliable model overall.
- `NumPy` CNN, though written from scratch, followed with strong recall and ROC AUC. Its success confirms the robustness of the custom implementation despite a lengthy inference time.
- Classical models that are trained and tested on flattened inputs, simply cannot capture spatial patterns

### 2. Scratch vs framework

Interestingly, the `CuPy` CNN's performance was almost on par with the `PyTorch` version. This is despite being trained using a manual backprop pipeline and custom operations, however comes at the cost of engineering time.

### 3. Threshold behavior matters

In our FNR-FPR analysis, CNNs maintained better FNR–FPR balance. Logistic Regression showed a more significant drop-off when moving away from its `FNR = FPR` threshold which reinforces that simple models may break down in uncertain operating ranges. CNNs are simply more accurate and more stable across decision boundaries.

### 4. Justice for classical models!!

Although underperforming overall, Random Forest was surprisingly competitive:

- Exceeded Logistic Regression by ~11% in accuracy and had a significantly better AUC (0.86 vs. 0.71) while not being far behind the deep learning models

- More memory efficient than both CNNs

- May be a strong choice for fast and efficient baseline filtering

## Conclusion + Learnings

At the start of this project, I expected deep learning to outperform everything else (and it did), but I didn’t expect how much infrastructure matters. The `PyTorch` model trained in minutes and gave great results. My own `NumPy`/`CuPy` implementation took far longer to run and debug, but seeing the training loop finish and save the model is an unmatched feeling. I was able to apply CNNs in a context I was interested in, that too at a much deeper level than just calling `model.fit()`.

I also grasped how how performance metrics don’t tell the whole story. Criterion like inference time, threshold sensitivity, and implementation complexity were something I hadn't considered in the past; they shape whether a model is usable and deployable, especially in the context of a large scale sky survey.

Finally, I have a **much stronger appreciation** for the original BRAAI paper. This was the first research paper I've read through and studied thorougly, and even though I didn’t copy their model exactly, I now understand how much thought went into their decisions and the complementary tools they built for astrophysical research.

I started this as a machine learning exercise. I finished it with a significantly better understanding of what it takes to turn a whiteboard design to code. If you have any feedback, questions, or comments, always feel free to reach out.

Parth Kotwal ❤️
