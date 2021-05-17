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

df = df.sample(frac=1).reset_index(drop=True)
train_part = 0.8
train_index = int(len(df) * train_part)
train_df = df.loc[0: train_index].reset_index(drop=True)
test_df = df.loc[train_index: len(df)].reset_index(drop=True)

supernode_features = [manufacturing_region]
features_list = [customer_industry, customer_region, product_family, make_vs_buy, ordered_qty_bucket, new_old_customer]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable)
super_tree.fit(train_df)
results, failed = evaluate(super_tree, test_df)

problematic_count = len([r for r in results.values() if r.segment.is_problematic])
total_count = len(results)
print(f"problematic segment count: {problematic_count} out of {total_count}")

print(f"failed count: {len(failed)} out of {len(test_df)}")
for segment_id, result in results.items():
    train_rows = result.train_rows
    train_perc = train_rows / len(train_df) * 100
    test_rows = result.rows
    test_perc = test_rows / len(test_df) * 100
    diff_perc = abs(train_perc - test_perc)
    distr = [v / sum(result.class_counts) if sum(result.class_counts) != 0 else 0 for v in result.class_counts]
    diff_distr = sum([(v - 0.2) ** 2 for v in distr])
    distr_str = ', '.join(["{:.2f}".format(float(c)) for c in distr])
    cutoffs_str = ', '.join(["{:.2f}".format(float(c)) for c in result.segment.gm_cutoffs])
    values = result.segment.segment_pairs
    print(f"train: {train_rows} ({train_perc:.2f}%), test: {result.rows} ({test_perc:.2f}%), diff: {diff_perc:.5f}, diff_distr: {diff_distr:.5f}, distr: [{distr_str}], cutoffs: [{cutoffs_str}]")
