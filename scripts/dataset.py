import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Example purchase data
purchase_data = {
    "person_id": [101, 101, 101, 202, 202, 202, 202,
                  202, 303, 303, 303, 404, 404, 404,],
    "date": ["2024-01-01", "2024-01-01", "2024-01-01",
             "2024-01-02", "2024-01-02", "2024-01-03",
             "2024-01-03", "2024-01-03", "2024-01-04",
             "2024-01-04", "2024-01-04", "2024-01-05",
             "2024-01-05", "2024-01-05", ],
    "Combined category": ["Pizza", "Soft Drink", "Salad",
                          "Burger", "Fries", "Pizza",
                          "Chicken Wings", "Soft Drink",
                          "Pasta", "Wine", "Garlic Bread",
                          "Burger", "Soft Drink", "Ice Cream", ]
}
purchase_df = pd.DataFrame(purchase_data)

# Create transactions from purchase data
transactions = (
    purchase_df
    .groupby(
        ['person_id', 'date']
    )['Combined category']
    .apply(list).tolist()
)
# Initialize transaction encoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

# Convert to DataFrame
df_te = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm with a default minimum support threshold (0.5)
frequent_itemsets = apriori(df_te, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)

# Create DataFrame for nodes
items = set()
for itemset in frequent_itemsets['itemsets']:
    items.update(itemset)
nodes = pd.DataFrame(list(items), columns=["Id"])

# Create DataFrame for edges
edges = rules[['antecedents', 'consequents', 'support',
               'confidence', 'lift']].copy()
edges['Source'] = edges['antecedents'].apply(lambda x: ', '.join(list(x)))
edges['Target'] = edges['consequents'].apply(lambda x: ', '.join(list(x)))
edges = edges.drop(columns=['antecedents', 'consequents'])

# Save files to CSV
nodes.to_csv("nodes.csv", index=False)
edges.to_csv("edges.csv", index=False)
