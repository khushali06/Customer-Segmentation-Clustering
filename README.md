# Customer Segmentation with Machine Learning (Clustering)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Visualization-Plotly%20%7C%20Seaborn-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project implements an end-to-end unsupervised machine learning pipeline to segment mall customers based on their demographic details, financial profiles, and purchasing behaviors. 

By employing **K-Means Clustering** and validating findings via **Hierarchical Clustering (Dendrogram analysis)**, this notebook extracts multi-dimensional behavioral personas. These data-driven clusters allow business and marketing teams to optimize promotional spending, tailor promotional messaging, and maximize customer lifetime value (CLV).

---

## Key Features & Highlights

- **Multi-Dimensional Exploratory Data Analysis (EDA):** Univariate distributions, bivariate pair plots, and interactive 3D feature spaces ($Age \times Income \times Spending$) built with Plotly.
- **Multiple Segmentation Strategies:**
  1. **Primary Model (Income vs. Spending Score):** Pinpoints core revenue segments and budget categories.
  2. **Demographic Model (Age vs. Spending Score):** Detects generational spending behaviors.
  3. **Categorical Model (Gender vs. Spending Score):** Highlights consumption differences across genders.
  4. **Engineered Metric Model (Age vs. Spending Propensity):** Evaluates spending efficiency relative to income ($\text{Score} / \text{Annual Income}$).
- **Cluster Optimization & Validation:** Determining optimal clusters ($k$) using the **Elbow Method (Within-Cluster Sum of Squares - WCSS)** and corroborating via **Hierarchical Dendrograms** (Ward's minimum variance criterion).
- **Data-Driven Persona Mapping:** Detailed quantitative breakdowns and targeted commercial strategies for each cluster.

---

## Dataset Description

The dataset contains transactional and demographic records of mall customers:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | Numerical / ID | Unique identifier for each shopper (dropped during preprocessing) |
| `Gender` | Categorical | Binary category (`Male`, `Female`) |
| `Age` | Numerical | Age of the customer in years (range: 18 – 70) |
| `Annual Income (k$)` | Numerical | Estimated annual earnings in thousands of USD (range: 15k – 137k) |
| `Spending Score (1-100)` | Numerical | Score assigned based on customer behavior and purchasing frequency |
| `Spending_Propensity` | Engineered | Ratio: $\frac{\text{Spending Score}}{\text{Annual Income} + 10^{-6}}$ |

---


## Model Experiments & Insights

### Model 1: Annual Income vs. Spending Score ($k = 5$)
The Elbow Method and Ward Dendrogram both indicate an optimal break at $k = 5$.

| Cluster | Persona Title | Avg. Age | Avg. Income ($k) | Avg. Spending (1-100) | Size | Actionable Business Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **Standard / Middle Class** | ~42.7 | ~55.3 | ~49.5 | 81 | Maintain retention with regular engagement and loyalty rewards. |
| **1** | **Target / High-Rollers** | ~32.7 | ~86.5 | ~82.1 | 39 | Primary target for premium services, luxury arrivals, and exclusive VIP programs. |
| **2** | **Young Enthusiasts** | ~25.3 | ~25.7 | ~79.4 | 22 | Capitalize on trendy merchandise, experiential marketing, and student discounts. |
| **3** | **Careful High-Earners** | ~41.1 | ~88.2 | ~17.1 | 35 | Target with value-focused, high-utility, and investment-grade promotions. |
| **4** | **Budget Shoppers** | ~45.2 | ~26.3 | ~20.9 | 23 | Offer budget-friendly bundles, clearance sales, and price-saving promotions. |

---

### Model 2: Age vs. Spending Score ($k = 4$)
- **Young High-Spenders ($\le 40$ yrs, Score $> 60$):** High reactivity to seasonal trends and digital campaigns.
- **Young Moderate/Low Spenders ($\le 40$ yrs, Score $\le 60$):** Price-conscious buyers entering the workforce.
- **Mature Moderate Spenders ($> 40$ yrs, Score 35–60):** Steady, predictable household-oriented purchasing patterns.
- **Senior Frugal Shoppers ($> 40$ yrs, Score $< 35$):** Infrequent shoppers requiring targeted clearance or utility offers.

---

### Model 3: Gender vs. Spending Score ($k = 4$)
- Female shoppers show slightly higher mean spending overall with denser clusters in the mid-to-high spending tiers.
- Male shoppers demonstrate higher dispersion, with polarized groups at extreme high and low ends.

---

### Model 4: Feature Engineered Model — Age vs. Spending Propensity ($k = 3$)
Using the engineered feature $\text{Propensity} = \frac{\text{Spending Score}}{\text{Income}}$:
- **Younger, High Propensity:** High willingness to spend despite low baseline income (trend seekers).
- **Younger, Low Propensity:** Cautious younger demographics saving or constrained by expenses.
- **Older, Low Propensity:** Established assets and income, with calculated, low discretionary spending.

---

## Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer-segmentation-clustering.git
   cd customer-segmentation-clustering
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/customer_segmentation.ipynb
   ```

---

## Dependencies

- Python 3.10+
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plotly.com/python/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [SciPy](https://scipy.org/)

---



### Summary of what was done:
- Synthesized all analytical steps from your notebook: setup, statistical checks, EDA (univariate, bivariate pairplot, 3D Plotly visualization), clustering workflows (Models 1 through 4), and hierarchical dendrogram analysis.
- Formatted business personas into actionable tables suitable for a machine learning portfolio.
- Structured mathematical formulations (Standardization, WCSS, Ward's method) with proper formatting and organized clean setup commands for repository visitors.
