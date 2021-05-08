from dataframe import *
from chaid import SuperCHAID

supernode_features = [manufacturing_region, product_family]
features_list = [customer_region, intercompany, make_vs_buy]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable)
super_tree.fit(df)

input_row = df.loc[0]
print(input_row[supernode_features + features_list])
print()

segment = super_tree.predict(input_row)
print("Supernode pairs:", segment.supernode_pairs)
print("Segment pairs:", segment.segment_pairs)
print(segment)
