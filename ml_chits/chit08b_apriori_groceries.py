# Chit 8 (Set B): Association Rule Learning using Apriori - Groceries Dataset
# pip install mlxtend

# 1. Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
print("Libraries imported successfully!")

# 2. Load Dataset
data = pd.read_csv("Groceries_dataset.csv")
print("Dataset loaded successfully!")
print("Shape:", data.shape)
print("\nFirst 5 Records:")
print(data.head())

# 3. Check Required Columns
print("\nColumn Names:")
print(data.columns.tolist())
data.columns = data.columns.str.strip()
data["Member_number"] = data["Member_number"].astype(str)
data["Date"] = data["Date"].astype(str)
data["itemDescription"] = data["itemDescription"].astype(str)
data = data.dropna(subset=["Member_number", "Date", "itemDescription"])
print("\nMissing values removed.")

# 4. Create Transactions
transactions = (
    data.groupby(["Member_number", "Date"])["itemDescription"]
    .apply(list)
    .tolist()
)
print("\nTotal Transactions:", len(transactions))
print("\nFirst 10 Transactions:")
for i, transaction in enumerate(transactions[:10], 1):
    print(i, transaction)

# 5. Transaction Encoding
te = TransactionEncoder()
encoded_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(encoded_array, columns=te.columns_)
transaction_df = transaction_df.astype(bool)
print("\nOne-Hot Encoded Dataset:")
print(transaction_df.head())

# 6. Apply Apriori Algorithm
# Note: with min_support = 0.01 this file gives no rules (baskets are small),
# so 0.001 is used.
frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
print("\nFREQUENT ITEMSETS")
print(frequent_itemsets.head(20).to_string(index=False))

# 7. Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.10)
print("\nTotal Association Rules:", len(rules))

# 8. Select Important Measures
rules_result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()

# 9. Convert Itemsets into Readable Text
rules_result["antecedents"] = rules_result["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
rules_result["consequents"] = rules_result["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

# 10. Round Values
rules_result["support"] = rules_result["support"].round(4)
rules_result["confidence"] = rules_result["confidence"].round(4)
rules_result["lift"] = rules_result["lift"].round(4)

# 11. Sort by Lift
rules_result = rules_result.sort_values(by="lift", ascending=False)

# 12. Display Association Rules
print("\nTOP 20 ASSOCIATION RULES")
print(rules_result.head(20).to_string(index=False))

# 13. Strong Association Rules
strong_rules = rules_result[(rules_result["confidence"] >= 0.20) & (rules_result["lift"] > 1)]
print("\nSTRONG ASSOCIATION RULES")
print(strong_rules.head(20).to_string(index=False))

# 14. Best Association Rule
if len(rules_result) > 0:
    best_rule = rules_result.iloc[0]
    print("\nBEST ASSOCIATION RULE")
    print("Antecedent :", best_rule["antecedents"])
    print("Consequent :", best_rule["consequents"])
    print("Support :", best_rule["support"])
    print("Confidence :", best_rule["confidence"])
    print("Lift :", best_rule["lift"])
