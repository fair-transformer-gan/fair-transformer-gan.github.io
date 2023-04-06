---
title: About
layout: default
nav_order: 2
---

## About
### A 2023 UC Berkeley Capstone project to mitigate racial, gender and other forms of bias in data and machine learning models

To address the potential biases in machine learning models that are increasingly present in our daily lives, we created a model that generates synthetic training data to mitigate various forms of bias in datasets. Our approach advances the state-of-the-art FairGAN+ by incorporating a transformer architecture, resulting in more accurate and fairer data generation. These modifications have the potential to improve the quality and equity of synthetic data, which is crucial in various applications, such as data augmentation and privacy-preserving machine learning.

#### Model Architecture
Fairness and accuracy are critical factors in generating high-quality synthetic data for various applications in machine learning. To address these challenges, we made modifications to the generator architecture of FairGAN+. By leveraging the transformer architecture, we were able to generate more accurate and fair synthetic data. 

##### Transformer
We integrated a transformer into the original generator architecture to capture the collinearity relationship between the features, outcome variable, and protected attribute. The attention mechanism was incorporated into the output of the FairGAN+ generator encoder to generate contextualized embeddings. These embeddings were then fed into the FairGAN+ generator decoder to produce continuous-value synthetic data.

{: .highlight }
Add more on Architecture

#### Aknowledgements
We would like to acknowledge the influential work of Depeng Xu, Shuhan Yuan, Lu Zhang, and Xintao Wu from the University of Arkansas, whose research on [FairGAN+: Achieving Fair Data Generation and Classification through Generative Adversarial Nets] greatly influenced our project.


[FairGAN+: Achieving Fair Data Generation and Classification through Generative Adversarial Nets]: https://ieeexplore.ieee.org/abstract/document/9006322?casa_token=rtdWVzSgLKoAAAAA:AMi_jcLYpcU-evETPjOU7z-NF7W6NVOBczeq01sPpEIzl8V_XcwMYeTqabxFM2AOwCYt2VA
