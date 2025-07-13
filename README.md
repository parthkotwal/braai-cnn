# `braai-cnn`: Real/Bogus Transient Detection using CNNs and Classical ML

This is a deep learning project focused on distinguishing true astrophysical events from false detections in Zwicky Transient Facility (ZTF) imaging data. The project reimplements the BRAAI CNN in both `PyTorch` and `CuPy` (GPU-accelerated `NumPy`) and evaluates them against classical machine learning models such as Logistic Regression and Random Forest classifiers.

## Objective
My goal was to push myself beyond classical ML approaches and venture into an untapped land-**deep learning**. 
```
<audio controls>
<source src="spooky_sound.mp3">
</audio>
```
Building off of [a previous astrophysics + ML project](https://github.com/parthkotwal/Star-Class-Forest), I wanted to continue the trend of mastering these skills using data I actually understood and cared about. This time, it was distinguishing real astrophysical events from false positives - a deceptively simple issue that's actually central to time domain astronomy.

This project was also a great learning exercise. Beyond implementing models and evaluating them, understanding the "why" behind their behavior:
- Why does preserving spatial structure matter in classification?
- Why does model architecture shape what the model learns and misses?
- Why do some regions of the dataset end up with oddly specific weights and biases?
