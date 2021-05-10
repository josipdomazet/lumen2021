from dataframe import *
from chaid import SuperCHAID

supernode_features = [manufacturing_region, product_family]
features_list = [customer_region, intercompany, make_vs_buy]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable)
super_tree.fit(df)

input_row = df.loc[11]
input_row[make_vs_buy] = None
print(input_row[supernode_features + features_list])
print()

segment, segment_pairs, imputed_pairs = super_tree.predict(input_row, impute=True)
print("Imputed pairs:", imputed_pairs)
print("Supernode pairs:", segment.supernode_pairs)
print("Segment pairs:", segment_pairs)
print(segment)

i = 0
for _, tree in super_tree.trees.items():
    tree.render(str(i))
    i+=1
