import numpy as np
import pandas as pd

from CHAID import Tree


class Segment:
    
    def __init__(self, leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs):
        self.leaf = leaf
        self.supernode_pairs = supernode_pairs
        self.segment_pairs = segment_pairs
        self.segment_df = segment_df
        self.gm_cutoffs = gm_cutoffs
        self.duplicates = self.gm_cutoffs.shape[0] < 6
        
    @property
    def is_problematic(self):
        return self.duplicates
        
    @property
    def rows_count(self):
        return len(self.segment_df)

    def __str__(self):
        cutoffs = ', '.join(f"{v:.3f}" for v in self.gm_cutoffs)
        return f"{self.segment_pairs}, rows: {self.rows_count}, cutoffs: {cutoffs}, problematic: {self.is_problematic}"


class SuperCHAID:

    SINGLETON_KEY = ("", )

    def __init__(self, supernode_features, features_list, dependant_variable, verbose=True,
                 alpha_merge=0.05, max_depth=2, 
                 min_parent_node_size=30, min_child_node_size=30,
                 split_threshold=0, is_exhaustive=False):
        self.supernode_features = supernode_features
        self.features_list = features_list
        self.dependant_variable = dependant_variable
        self.verbose = verbose
        self.alpha_merge = alpha_merge
        self.max_depth = max_depth
        self.min_parent_node_size = min_parent_node_size
        self.min_child_node_size = min_child_node_size
        self.split_threshold = split_threshold
        self.is_exhaustive = is_exhaustive
        
    def fit(self, df):
        self.trees = {}
        
        for supernode_values in self._determine_supernode_values_list(df):
            supernode_df = df
            supernode_pairs = dict(zip(self.supernode_features, supernode_values))
            
            for i, supernode_feature in enumerate(self.supernode_features):
                supernode_df = supernode_df[supernode_df[supernode_feature] == supernode_values[i]]
            supernode_df = supernode_df.reset_index(drop=True)
            if len(supernode_df) == 0: continue

            tree = self._fit_tree(supernode_df)
            tree.supernode_pairs = supernode_pairs
            tree.supernode_df = supernode_df
            tree.segments = []
            
            for leaf in tree.classification_rules():
                segment_df = supernode_df
                segment_pairs = {}
                
                for variable_data_pair in leaf['rules']:
                    variable = variable_data_pair['variable']
                    data = variable_data_pair['data']
                    segment_pairs[variable] = data
                    filter = np.zeros(len(segment_df), dtype=bool)
                    for value in data:
                        filter |= segment_df[variable] == value
                    segment_df = segment_df[filter].reset_index(drop=True)
                    
                gm_cutoffs = self._determine_gm_cutoffs(segment_df)
                segment = Segment(leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs)
                tree.segments.append(segment)
                
            self.trees[supernode_values] = tree
            
            if self.verbose:
                print(f"Finished tree: {supernode_pairs}.")
                print(f"  Rows: {len(supernode_df)}")
                print(f"  Segments: {len(tree.segments)}")
                for segment in tree.segments:
                    print(f"    {segment}")
                print()
            
    def predict(self, input_row):
        if self.is_singleton:
            tree = self.trees[SuperCHAID.SINGLETON_KEY]
        else:
            key = []
            for supernode_feature in self.supernode_features:
                key.append(input_row[supernode_feature])
            tree = self.trees[tuple(key)]
            
        for segment in tree.segments:
            if self._belongs_to_segment(input_row, segment):
                return segment

    @property
    def is_singleton(self):
        return len(self.supernode_features) == 0
    
    @property
    def singleton(self):
        return self.trees[SuperCHAID.SINGLETON_KEY]

    def _determine_supernode_values_list(self, df):
        if self.is_singleton: return [SuperCHAID.SINGLETON_KEY]

        supernode_values_list = []
        history = set()
        for v in df[self.supernode_features].values:
            v = tuple(v)
            if v not in history:
                history.add(v)
                supernode_values_list.append(v)
        return supernode_values_list
        
    def _fit_tree(self, supernode_df):
        return Tree.from_pandas_df(
          supernode_df,
          dict(zip(self.features_list, ["nominal"] * len(self.features_list))),
          self.dependant_variable,
          dep_variable_type="continuous",
          alpha_merge=self.alpha_merge,
          max_depth=self.max_depth,
          min_parent_node_size=self.min_parent_node_size,
          min_child_node_size=self.min_child_node_size,
          split_threshold=self.split_threshold,
          is_exhaustive=self.is_exhaustive
        )
        
    def _determine_gm_cutoffs(self, segment_df):
        segment_df_dependant_variable = segment_df[self.dependant_variable].sort_values()
        _, bins = pd.qcut(segment_df_dependant_variable, q=[0, .2, .4, .6, .8, 1], retbins=True, duplicates="drop")
        return bins
        
    def _belongs_to_segment(self, input, segment):
        for key, values in segment.segment_pairs.items():
            if input[key] not in values: return False
        return True