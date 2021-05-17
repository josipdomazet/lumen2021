import numpy as np
import pandas as pd
from graphviz import Digraph

from CHAID import Tree


class Segment:
    
    def __init__(self, segment_id, leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs):
        self.segment_id = segment_id
        self.leaf = leaf
        self.supernode_pairs = supernode_pairs
        self.segment_pairs = segment_pairs
        self.segment_df = segment_df
        self.gm_cutoffs = gm_cutoffs
        self.duplicates = len(self.gm_cutoffs) < 6
        
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
                 alpha_merge=0.1, max_depth=4,
                 min_parent_node_size=800, min_child_node_size=800,
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
        self.id_counter = 0
        
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
                segment = self._create_segment(leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs)
                tree.segments.append(segment)
                
            tree.root = self._rebuild(tree, supernode_df, {}, tree.tree_store[0])
            self.trees[supernode_values] = tree
            
            if self.verbose:
                print(f"Finished tree: {supernode_pairs}.")
                print(f"  Rows: {len(supernode_df)}")
                print(f"  Segments: {len(tree.segments)}")
                for segment in tree.segments:
                    print(f"    {segment}")
                print()

    def predict(self, input_row, impute=True):
        input_row = input_row.copy()
        tree = self._get_tree(input_row)
        if tree is None: return None
        
        segment_pairs = {}
        imputed_pairs = {}
        current_node = tree.root
        
        while not current_node.is_terminal:
            variable = current_node.split.column
            value = input_row[variable] if impute else self._get_value(input_row, variable)

            for child_id, child in current_node.children.items():
                if value in child.choices:
                    current_node = child
                    break
            else:
                if not impute: break
                
                max_child = max(current_node.children.values(), key=lambda c: len(c.df))
                max_value = max(max_child.choices, key=lambda v: sum(max_child.df[variable] == v))
                value = max_value
                imputed_pairs[variable] = value
                input_row[variable] = value
            segment_pairs[variable] = value

        for segment in tree.segments:
            if self._belongs_to_segment(input_row, segment):
                return segment, segment_pairs, imputed_pairs
                
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
        
    def _create_segment(self, leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs):
        segment_id = self.id_counter
        self.id_counter += 1
        return Segment(segment_id, leaf, supernode_pairs, segment_pairs, segment_df, gm_cutoffs)

    def _rebuild(self, tree, df, pairs, node):
        node.df = df
        node.children = {}
        node.pairs = pairs
        
        variable = node.split.column
        for child in [c for c in tree.tree_store if c.parent == node.node_id]:
            values = child.choices
            filter = np.zeros(len(df), dtype=bool)
            for value in values:
                filter |= df[variable] == value
            new_df = df[filter].reset_index(drop=True)
            pairs = pairs.copy()
            pairs[variable] = values
            child.value = ', '.join([str(v) for v in values])
            node.children[child.node_id] = self._rebuild(tree, new_df, pairs, child)
        return node
        
    def _get_tree(self, input_row):
        if self.is_singleton:
            tree = self.singleton
        else:
            key = []
            for supernode_feature in self.supernode_features:
                key.append(input_row[supernode_feature])
            key = tuple(key)
            tree = self.trees[key] if key in self.trees else None
        return tree
        
    def _determine_gm_cutoffs(self, segment_df, impute=True):
        segment_df_dependant_variable = segment_df[self.dependant_variable].sort_values()
        _, bins = pd.qcut(segment_df_dependant_variable, q=[0, .2, .4, .6, .8, 1], retbins=True, duplicates="drop")
        bins = list(bins)
        if impute:
            while len(bins) < 6:
                assert len(bins) > 1, "Cannot impute GM class bound if length is less than 2."
                max_index = 0
                max_range = -1
                for i in range(len(bins)-1):
                    current_range = bins[i+1] - bins[i]
                    if current_range > max_range:
                        max_range = current_range
                        max_index = i
                bins.insert(max_index+1, (bins[max_index] + bins[max_index+1]) / 2)
        return bins
        
    def _belongs_to_segment(self, input_row, segment):
        for key, values in segment.segment_pairs.items():
            if self._get_value(input_row, key) not in values: return False
        return True
        
    def _get_value(self, input_row, key):
        value = input_row[key]
        return value if str(value) != "nan" else '<missing>'
        
        
class SuperCHAIDVisualizer:

    def __init__(self, super_tree, format="png",
                 internal_color="#BBBB00", leaf_color="#008800",
                 zero_entropy_color="#00CC00", problematic_color="#CC0000",
                 edge_label_color="#0000BB"):
        self.super_tree = super_tree
        self.format = format
        self.internal_color = internal_color
        self.leaf_color = leaf_color
        self.zero_entropy_color = zero_entropy_color
        self.problematic_color = problematic_color
        self.edge_label_color = edge_label_color
        self.dot = None
    
    def export(self, path):
        for i, (supernode_values, tree) in enumerate(self.super_tree.trees.items()):
            self.dot = Digraph(format=self.format)
            self._traverse(tree, tree.root)
            self.dot.render(f"{path}-{i+1}")
        
    def _adapt_segment_str(self, segment):
        cutoffs = ', '.join(f"{v:.3f}" for v in segment.gm_cutoffs)
        return f"id:{segment.leaf['node']}, rows: {segment.rows_count}, cutoffs:\n{cutoffs}".replace(", ", "\n")
        
    def _adapt_node_str(self, node):
        return f"id: {node.node_id}, feature: {node.split.column}, rows: {len(node.df)}".replace(", ", "\n")
        
    def _create_node(self, tree, node):
        if node.is_terminal:
            segment = [s for s in tree.segments if s.leaf['node'] == node.node_id][0]
            color = self.problematic_color if segment.is_problematic else self.leaf_color
            self.dot.node(str(node.node_id), self._adapt_segment_str(segment), style="filled", color=color)
        else:
            color = self.internal_color
            self.dot.node(str(node.node_id), self._adapt_node_str(node), style="filled", color=color)
        
    def _create_edge(self, from_id, to_id, label):
        self.dot.edge(str(from_id), str(to_id), label, fontcolor=self.edge_label_color)
        
    def _traverse(self, tree, node):
        self._create_node(tree, node)
        for child_id, child in node.children.items():
            feature_value = child.value
            self._create_node(tree, child)
            self._create_edge(node.node_id, child.node_id, f"{feature_value}")
            self._traverse(tree, child)
