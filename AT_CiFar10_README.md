# CIFAR-10 Reflection and Summary

### MNIST vs. CIFAR-10
---
###

The obvious distinction between MNIST and CIFAR-10 is the data they take in; a 28x28 pixel black and white image of a number vs. a 32x32 pixel RGB image of one of ten different animals/objects. The primary difference I noticed between MNIST and CIFAR-10 (in performance) is that CIFAR-10 takes significantly longer to train and is significantly less accurate than MNIST due to the nature of the data it must deal with.

---


### Ideas for Future Improvemnt
---

Some ways to potentially improve on the model in the future include:

1. Further optimization

2. Utilize the computer's GPU
    1. This will help with faster training and model optimization
3. Use new python libraries to randomly augment the image
    1. This will help prevent overfitting
    2. Will effectively increase the size of the dataset
    3. Will effectively make it so that the computer never sees the same image twice.
---
### How the Model Performs
---

![Performance graph]("C:\Users\ateddy624\Figure_1.png")