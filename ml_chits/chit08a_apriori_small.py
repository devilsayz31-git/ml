# Chit 8 (Set A): Association Rule Learning using Apriori on the given 5 transactions
# pip install mlxtend

# 1. Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 2. Create Transactions
transactions = [
    ["Bread", "Butter", "Jam", "Juice", "Milk"],   # T1
    ["Jam", "Juice", "Bread", "Butter"],           # T2
    ["Juice", "Milk", "Bread"],                    # T3
    ["Butter", "Juice", "Biscuit", "Milk"],        # T4
    ["Bread", "Milk", "Butter"],                   # T5
]
print("Total Transactions:", len(transactions))
for i, transaction in enumerate(transactions, 1):
    print(i, transaction)

# 3. Transaction Encoding
te = TransactionEncoder()
encoded_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(encoded_array, columns=te.columns_).astype(bool)
print("\nOne-Hot Encoded Dataset:")
print(transaction_df)

# 4. Apply Apriori Algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.4, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
print("\nFREQUENT ITEMSETS")
print(frequent_itemsets.to_string(index=False))

# 5. Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nTotal Association Rules:", len(rules))

# 6. Select Important Measures and make them readable
rules_result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
rules_result["antecedents"] = rules_result["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
rules_result["consequents"] = rules_result["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
rules_result = rules_result.round(4).sort_values(by="lift", ascending=False)

# 7. Display Association Rules
print("\nASSOCIATION RULES")
print(rules_result.to_string(index=False))

# 8. Strong Association Rules
strong_rules = rules_result[(rules_result["confidence"] >= 0.8) & (rules_result["lift"] > 1)]
print("\nSTRONG ASSOCIATION RULES")
print(strong_rules.to_string(index=False))

# 9. Best Association Rule
best_rule = rules_result.iloc[0]
print("\nBEST ASSOCIATION RULE")
print("Antecedent :", best_rule["antecedents"])
print("Consequent :", best_rule["consequents"])
print("Support :", best_rule["support"])
print("Confidence :", best_rule["confidence"])
print("Lift :", best_rule["lift"])
