class Result:

    def __init__(self, segment): 
        self.segment = segment
        self.rows = 0
        self.class_counts = [0] * (segment.gm_cutoffs.shape[0] - 1)
    
    @property
    def train_rows(self):
        return len(self.segment.segment_df)
        
    def __str__(self):
        return f"segment: {self.segment}\n rows: {self.rows}, counts: {self.class_counts}"


def evaluate(super_tree, test_df, gm_column_name="gm"):
    results = {}
    failed = []
    
    for i in range(len(test_df)):
        input_row = test_df.loc[i]
        prediction = super_tree.predict(input_row, impute=True)
        if prediction is None:
            failed.append(i)
            continue

        predicted_segment, _, _ = prediction
        predicted_segment_id = predicted_segment.segment_id
        if predicted_segment_id not in results:
            results[predicted_segment_id] = Result(predicted_segment)
        
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

supernode_features = []
features_list = [manufacturing_region, product_family, customer_region, top_customer_group]
dependant_variable = gm

super_tree = SuperCHAID(supernode_features, features_list, dependant_variable)
super_tree.fit(train_df)

results, failed = evaluate(super_tree, test_df)
print(f"failed count: {len(failed)} out of {len(test_df)}")
for segment_id, result in results.items():
    train_rows = result.train_rows
    train_perc = train_rows / len(train_df) * 100
    test_rows = result.rows
    test_perc = test_rows / len(test_df) * 100
    diff_perc = abs(train_perc - test_perc)
    cutoffs = ', '.join(["{:.2f}".format(float(c)) for c in result.segment.gm_cutoffs])
    values = result.segment.segment_pairs
    print(f"train: {train_rows} ({train_perc:.2f}%), test: {result.rows} ({test_perc:.2f}%), diff: {diff_perc:.2f}%, counts: {result.class_counts}, cutoffs: [{cutoffs}]")
