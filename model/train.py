import pickle


class Result:

    def __init__(self, segment): 
        self.segment = segment
        self.rows = 0
        self.class_counts = [0] * (len(segment.gm_cutoffs) - 1)
    
    @property
    def train_rows(self):
        return len(self.segment.segment_df)
        
    def __str__(self):
        return f"segment: {self.segment}\n rows: {self.rows}, counts: {self.class_counts}"


def evaluate(super_tree, test_df, gm_column_name="gm"):
    results = {}
    failed = []
    
    for tree in super_tree.trees.values():
        for segment in tree.segments:
            results[segment.segment_id] = Result(segment)
    
    for i in range(len(test_df)):
        input_row = test_df.loc[i]
        prediction = super_tree.predict(input_row, impute=True)
        if prediction is None:
            failed.append(i)
            continue

        predicted_segment, _, _ = prediction
        predicted_segment_id = predicted_segment.segment_id        
        result = results[predicted_segment_id]
        result.rows += 1
        
        current_gm = input_row[gm_column_name]
        for i, cutoff in enumerate(predicted_segment.gm_cutoffs):
            if current_gm < cutoff:
                result.class_counts[i-1] += 1
                break

    return results, failed


from dataframe import *
from chaid import SuperCHAID

supernode_features = [manufacturing_region]
features_list = [customer_industry, customer_region, product_family, make_vs_buy, ordered_qty_bucket, new_old_customer]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable, verbose=False)
super_tree.fit(df)

with open('chaid.model', 'wb') as handle:
    pickle.dump(super_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
