---
title: Results
layout: home
nav_order: 3
---

# Results
A high level overview of results from our experiments. In general, we find there is a statistically significant improvement in fairness (BER) depending on the dataset, but find there is a trade-off in data utility (f1-score).

## Balanced Error Rate (BER)
**Purpose:** Can we predict the protected attribute Z from X features (higher = better -> minimizes disparate impact)?

**Takeaway:**  Fair Transformer GAN’s generated data has statistically significant fairness improvements over SOTA generated and/or original data on most datasets.

![](/images/fairness_metric_ACCESSIBLE_viz.png)

## Utility Performance: Logistic Regression
**Purpose:** How accurate is the classifier when trained on generated data and predicted on original data?

**Takeaway:**  Fair Transformer GAN’s generated data results in lower F1 than original data (expected with a higher BER) and comparable F1 to SOTA generated data.

![](/images/log_reg_f1_metric_ACCESSIBLE_viz.png)

## Utility Performance: Random Forest
**Purpose:** How accurate is the classifier when trained on generated data and predicted on original data?

**Takeaway:**  Fair Transformer GAN’s generated data has comparable / lower F1 than original or SOTA generated data (expected with a higher BER).

![](/images/random_forest_f1_ACCESSIBLE_viz.png)


