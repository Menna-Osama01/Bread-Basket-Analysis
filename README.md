# Market Basket Analysis Project 🍞🛒

This project performs **Market Basket Analysis** on a retail dataset (`bread basket.csv`) to find frequent itemsets and generate association rules. The analysis includes data cleaning, preprocessing, custom Apriori implementation, rule generation, and visualizations.

---

## **Project Overview**

Market Basket Analysis helps understand customer purchase patterns. Using this project, you can:
- Identify **frequent items** bought together.
- Generate **association rules** to reveal relationships between products.
- Visualize frequent items, rules, and lift relationships.

---

## **Dataset**

- File: `bread basket.csv`
- Contains:
  - `Transaction` – unique transaction IDs
  - `Item` – product purchased
  - `date_time` – timestamp of purchase

The dataset is cleaned to remove:
- Invalid items (`none`, `nan`, empty strings)
- Duplicate transactions
- Non-standard separators

---

## **Dependencies**

Install required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn mlxtend networkx

How to Run the Project

Place the dataset bread basket.csv in the project folder.

Run the main Python script or Jupyter notebook (market_basket_analysis.ipynb).

The script will:

Clean the data

Convert transactions into a basket matrix (one-hot encoded)

Find frequent itemsets using a custom Apriori algorithm

Generate association rules

Save CSV outputs:

transactions.csv

basket.csv

freq_itemsets.csv

rules.csv

Generate plots:

project_top_items.png

project_support_confidence.png

project_support_lift.png

project_lift_heatmap.png

project_rules_graph.png

Key Functions
Custom Apriori

Finds frequent itemsets of any length.

Filters by minimum support.

Returns a DataFrame with itemsets and support.

Custom Association Rules

Generates rules from frequent itemsets.

Computes support, confidence, and lift.

Filters rules by thresholds.

Visualizations

Top Items – Bar chart of most frequent items.

Support vs Confidence – Scatter plot of all rules.

Support vs Lift – Scatter plot for rule importance.

Lift Heatmap – Heatmap of top 20 rules.

Rule Graph – Network graph of top association rules.

Usage Example

# Load basket
basket = pd.read_csv("basket.csv")

# Generate frequent itemsets
freq_itemsets = custom_apriori(basket, min_support=0.02, use_colnames=True)

# Generate rules
rules = custom_association_rules(freq_itemsets, metric="confidence", min_threshold=0.01)

# Save rules
rules.to_csv("rules.csv", index=False)

Project Output

CSV files for transactions, basket, frequent itemsets, and rules.

Visual plots to analyze patterns.

Strong rules for business insights.