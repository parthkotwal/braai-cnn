# `braai-cnn`: Real/Bogus Transient Detection using CNNs and Classical ML

This is a deep learning project focused on distinguishing true astrophysical events from false detections in Zwicky Transient Facility (ZTF) imaging data. The project reimplements the BRAAI CNN in both `PyTorch` and `CuPy` (GPU-accelerated `NumPy`) and evaluates them against classical machine learning models such as Logistic Regression and Random Forest classifiers.

## Objective
My goal was to push myself beyond classical ML approaches and venture into an untapped land-**deep learning**. 
```
<audio controls>
<source src="spooky_sound.mp3">
</audio>
```
Building off of [a previous astrophysics + ML project](https://github.com/parthkotwal/Star-Class-Forest), I wanted to continue the trend of mastering these skills using data I actually understood and was interested in. This time, it was distinguishing real astrophysical events from false positives - a deceptively simple issue that's actually central to time domain astronomy.

This project was also a great learning exercise. Beyond implementing models and evaluating them, understanding the "why" behind them. Why spatial patterns matter. Why the structure of a model affects what it learns and doesn't. Why sometimes the weirdest part of your dataset ends up having a quite peculiar set of weights and biases.
