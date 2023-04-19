---
title: Results
layout: home
nav_order: 3
---

# Results
A high level overview of results from our experiments. In general, we find there is a statistically significant improvement in fairness (BER) depending on the dataset, but find there is a trade-off in data utility (accuracy, f1-score).

## Balanced Error Rate (BER)
**Purpose:** Can we predict the protected attribute Z from X features (higher = better -> minimizes disparate impact)?
**Takeaway:**  Fair Transformer GAN’s generated data has statistically significant fairness improvements over SOTA generated and/or original data on most datasets.
![](/images/ber_results.png)

