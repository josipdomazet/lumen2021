from dataframe import *
from chaid import SuperCHAID, SuperCHAIDVisualizer

supernode_features = [manufacturing_region]
features_list = [customer_region, product_family, make_vs_buy]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable)
super_tree.fit(df)

visualizer = SuperCHAIDVisualizer(super_tree)
visualizer.export("tree")

input_row = df.loc[0]
input_row[make_vs_buy] = np.nan
print(input_row[supernode_features + features_list])
print()

result = super_tree.predict(input_row, impute=True)
if result is not None:
    segment, segment_pairs, imputed_pairs = result
    print("Imputed pairs:", imputed_pairs)
    print("Supernode pairs:", segment.supernode_pairs)
    print("Segment pairs:", segment_pairs)
    print(segment)
