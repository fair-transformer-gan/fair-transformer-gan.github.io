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

![](/images/ber_results.png)

## Utility Performance: Logistic Regression
**Purpose:** How accurate is the classifier when trained on generated data and predicted on original data?

**Takeaway:**  Fair Transformer GAN’s generated data results in lower F1 than original data (expected with a higher BER) and comparable F1 to SOTA generated data.

![](/images/lr_results.png)

## Utility Performance: Random Forest
**Purpose:** Technique of randomly increasing the outcome for the non-dominant classes to be similar to dominant class’s outcome.

**Takeaway:**  Fair Transformer GAN’s generated data has comparable / lower F1 than original or SOTA generated data (expected with a higher BER).

![](/images/rf_results.png)

## Upleveling
**Purpose:** How accurate is the classifier when trained on generated data and predicted on original data?

**Takeaway:** Higher fairness (15% randomness in our experiment), increases outcomes for everyone overall rather than down-leveling the original dominant class (as much), but higher false positive (lower acc).

**Note:** Upleveling is used for specific cases where it's important not to compromise any user group (e.g., healthcare)

![](/images/uplevel_results.png)
