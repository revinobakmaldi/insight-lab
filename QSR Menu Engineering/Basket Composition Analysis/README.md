# Case Study: Optimizing Menu Strategy with Market Basket Analysis  

Have you ever walked into a fast-food place and come out with only one item? Yeah, me neither. I usually grab a combo meal, then somehow convince myself I also need fries, maybe soup, and if I’m feeling adventurous, ice cream. One time I even bought a music CD from a band I’d never heard of. Why? Because apparently my stomach also wanted to feed my curiosity.

Now, my little shopping story doesn’t really matter on its own. But imagine collecting thousands of stories like mine. Suddenly, you don’t just have funny snack habits, you’ve got serious data. And that data can tell you things like which items love to hang out together on the receipt, or what combo could turn a regular meal into a “why did I just spend extra on dessert?” kind of order.

That’s what this case study is about: how I turned **piles of customer transactions** into actual **business strategies** that help companies sell smarter.

## 1. Project Overview & Business Problem

**The Company:** One of leading Quick Service Restaurant (QSR) chain with a global presence.

**The Business Problem:** The company lacked a data-driven understanding of which menu items customers purchased together. This made it difficult to:

- Design effective combo meals and promotions.
- Optimize menu layout (digital and physical) for upselling and cross-selling.
- Ultimately, increase the Average Order Value (AOV).

**My Role:** Lead Data Analyst, responsible for end-to-end analysis from data extraction to insight delivery.

**Project Goal:** To uncover significant product associations within customer transactions to inform menu strategy and marketing campaigns.

## 2. Technical Approach & Methodology

### A. Data Sources & Tools

**Tools Used:** Python, Jupyter Notebook, & Power BI.

**Python Libraries:**

```python
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
```

**Data Source:** Transaction-level sales data from the company's data warehouse.

**Key Attributes:** I worked with tables containing `order_id`, `product_id`, `product_name`, and `transaction_date`.

### B. Data Preprocessing

**Data Cleaning:** Filtered out invalid transactions.

**Data Transformation:** I grouped transactions by `order_id` to create baskets of items, then converted them into a one-hot encoded table. Each row represents an order, and each column indicates whether a product was included (True/False). This structure prepares the data for association rule mining.

```python
# Prepare transactions
# df is a dataframe contains transaction data from datawarehouse
transactions = df.groupby('order_id')['product_id'].apply(lambda x: sorted(set(x))).tolist()

# One-hot encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)
```

### C. Analysis Technique: Association Rule Mining

**Algorithm Selected:** I used the FP-Growth algorithm because it is faster and more efficient than Apriori for large datasets. Unlike Apriori, which generates and tests many candidate itemsets, FP-Growth compresses the dataset into a compact FP-tree and directly extracts frequent patterns. This reduces computational cost, avoids repeated database scans, and makes it well-suited for millions of QSR transactions.

**Metric Explanation:** I evaluated the strength of the rules between item sets using three key metrics:

- Support: How frequently an itemset appears in all transactions (i.e., its popularity).
- Confidence: The probability that a customer who bought item A will also buy item B.
- Lift: How much more likely a customer is to buy item B when they buy item A, compared to the likelihood of buying B randomly. A lift > 1 indicates a rule that is useful and unexpected.

**Parameter Tuning:** Iteratively tested different minimum support and confidence thresholds to find a balanced number of meaningful rules.

```python
# Set parameter
min_support = 0.01
metric = 'lift' 
min_threshold = 1

# Frequent itemsets
frequent_fp = fpgrowth(basket_df, min_support=min_support, use_colnames=True)

# Association rules
rules = association_rules(frequent_fp, metric=metric, min_threshold=min_threshold)

# Sort
rules = rules.sort_values(by=metric, ascending=False)
```

## 3. Key Findings & Visualization

Due to confidentiality, I am going to show only the flow & pricinples I used in this analysis, not the implementation using real data.

### **1. High-Lift Pairings**

- **Lift > 1** means the relationship is stronger than chance.
- The higher the lift, the more “surprising” and valuable the association.  
- *Example:* Fries → Soda with Lift = 3.5 means customers who buy fries are 3.5x more likely to also buy soda.  

### **2. Strong Confidence**

- Confidence shows how reliably item A leads to item B.  
- High confidence (>50–60%) means the rule has predictive strength.  
- *Example:* 70% of people who buy Burger also buy Fries.  

### **3. High-Support Core Items**

- Support shows how frequent an itemset is in the whole dataset.  
- High-support items are your **core products** that appear in many baskets (anchors for combos).  
- *Example:* Coffee shows up in 40% of all transactions.  

### **4. Low-Support but High-Lift Pairs**

- Rare items that strongly pair with another can signal **niche upselling opportunities**.  
- *Example:* Soup + Dessert, bought by only 5% of customers, but with Lift = 4.0.  

### **5. Cross-Category Opportunities**

- Look for rules that link different product categories.  
- *Example:* Chicken meals → Coffee (unexpected but valuable for cross-selling).  

### **6. Missed Opportunities (Negative Space)**

- Sometimes, popular items don’t pair with each other — spotting these gaps can inspire new promotions.  
- *Example:* Burgers rarely sold with Ice Cream → create a “Burger + Ice Cream” bundle to test.  

### **7. Actionability**  

- Don’t just look at metrics; ask *“Can the business use this?”*  
- Rules should inform **menu bundles, digital suggestions, store training, or targeted promotions**.  

Also, here some functions to create the visualization during my analysis

```python
# Plot Lift Heatmap
def plot_lift_heatmap(rules, figsize=(6, 5), fontsize=6):
    """
    Plots a heatmap of Lift values from association rules with custom font and size settings.

    Parameters:
        rules (pd.DataFrame): Output from association_rules().
        figsize (tuple): Size of the figure.
        fontsize (int): Base font size for all elements.
    """
    # Prepare data for heatmap
    matrix = rules.copy()
    matrix['antecedent'] = matrix['antecedents'].apply(lambda x: ', '.join(list(x)))
    matrix['consequent'] = matrix['consequents'].apply(lambda x: ', '.join(list(x)))
    pivot = matrix.pivot(index='antecedent', columns='consequent', values='lift')

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        annot_kws={"size": fontsize},
        cbar_kws={'label': 'Lift'}
    )
    ax.set_xlabel('Consequent Menu', fontsize=fontsize)
    ax.set_ylabel('Antecedent Menu', fontsize=fontsize)
    ax.set_title('Association Rule Lift Matrix', fontsize=fontsize + 2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Lift', fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.show()
```

```python
# Network Graph
def plot_rules_network(rules, figsize=(6, 5), node_font=8, edge_font=7, title_font=10):
    """
    Plots a directional network graph of association rules, removing bidirectional arrows
    by keeping only the stronger (higher-confidence) direction.
    """

    # Flatten antecedent and consequent to strings (assuming only 1 item each)
    rules = rules.copy()
    rules['a'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['b'] = rules['consequents'].apply(lambda x: list(x)[0])

    # Build dictionary to store strongest direction for each unordered pair
    best_rules = {}
    for _, row in rules.iterrows():
        a, b = row['a'], row['b']
        pair_key = tuple(sorted([a, b]))
        direction = (a, b)
        confidence = row['confidence']
        lift = row['lift']

        # If this pair not seen, or this direction has higher confidence → keep it
        if pair_key not in best_rules or confidence > best_rules[pair_key]['confidence']:
            best_rules[pair_key] = {
                'source': direction[0],
                'target': direction[1],
                'confidence': confidence,
                'lift': lift
            }

    # Build graph
    G = nx.DiGraph()
    for rule in best_rules.values():
        G.add_edge(rule['source'], rule['target'],
                   weight=rule['lift'],
                   label=f"{rule['confidence']:.2f}")
    pos = nx.spring_layout(G, k=0.6, seed=42)
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=300,
        font_size=node_font,
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1'
    )
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=edge_font)
    plt.title("Association Rules Network (Strongest Direction Only)", fontsize=title_font)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

```python
# Rules Scatter Plot
def plot_rules_scatter(rules, figsize=(6, 5), fontsize=6):
    """
    Plots a labeled scatterplot of association rules with support vs confidence,
    colored by lift. Labels are auto-generated from antecedents → consequents.
    
    Parameters:
        rules (pd.DataFrame): DataFrame from association_rules() with required columns.
        figsize (tuple): Figure size.
        fontsize (int): Font size for labels, ticks, and title.
    """
    # Auto-generate label column if not present
    rules = rules.copy()
    rules['label'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))) + \
                     " → " + \
                     rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # Setup plot layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Scatterplot
    ax = plt.subplot(gs[0])
    sc = ax.scatter(
        rules['support'],
        rules['confidence'],
        c=rules['lift'],
        cmap='viridis',
        alpha=0.7
    )

    # Text labels
    for _, row in rules.iterrows():
        ax.text(row['support'], row['confidence'], row['label'], fontsize=fontsize, alpha=0.8, zorder=10)

    # Axis & title
    ax.set_xlabel('Support', fontsize=fontsize)
    ax.set_ylabel('Confidence', fontsize=fontsize)
    ax.set_title('Association Rules Scatterplot', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Colorbar
    cax = plt.subplot(gs[1])
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Lift', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.show()
```

## 4. Impact & Business Recommendations

This is the "so what?" section. It shows you can translate data into action.

Strategic Impact: "The findings were presented to the Head of Marketing and Menu Strategy and directly influenced several business decisions:"

- Menu Bundle Design: "Informed the creation of new limited-time combo meals based on the high-lift, high-confidence pairings we identified."
- Digital Menu Optimization: "Guided the layout of the mobile app and kiosk menu to suggest relevant add-ons (e.g., 'Customers who bought this also enjoyed...') based on the rules."
- Staff Training: "The insights were distilled into simple upselling guidelines for crew members at the point of sale."

Generalized Outcome: "This data-driven approach to menu strategy replaced previous guesswork and legacy assumptions. The project was hailed as a success for providing a clear, analytical foundation for critical business choices aimed at boosting revenue per customer."

## 5. Lessons Learned & Next Steps

Challenges Overcome: "The main challenge was the computational complexity of processing millions of transactions. This was solved by optimizing the code and using sampling techniques for initial exploration."

Future Work: "A potential next step would be to segment the analysis by time of day (e.g., breakfast vs. dinner) or by customer segment to uncover even more nuanced patterns."

Personal Learning: "This project was an excellent practical application of unsupervised learning and taught me the importance of not just finding patterns, but contextualizing them for a non-technical business audience to drive action."
