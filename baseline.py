import math

import numpy as np
import pandas as pd
from graphviz import Digraph


class Segment:

    def __init__(self, leaf, feature_values, gm_cutoffs, zero_entropy=False):
        self.leaf = leaf
        self.feature_values = feature_values
        self.gm_cutoffs = gm_cutoffs
        self.duplicates = self.gm_cutoffs.shape[0] < 6
        self.zero_entropy = zero_entropy
        
    @property
    def is_problematic(self):
        return self.duplicates
        
    def __str__(self):
        cutoffs = '; '.join(["{:.2f}".format(float(c)) for c in self.gm_cutoffs])
        return f"id: {self.leaf.node_id}, rows: {self.leaf.rows_count}, feature: {self.leaf.feature_name}, cutoffs: [{cutoffs}], duplicates: {self.duplicates}, H == 0: {self.zero_entropy}"


class Node:
    
    def __init__(self, node_id, feature_name, rows_count):
        self.node_id = node_id
        self.feature_name = feature_name
        self.rows_count = rows_count
        self.children = {}

    @property
    def is_leaf(self):
        return len(self.children) == 0
        
    def print_subtree(self, depth=0, indent=" " * 4):
        print(f"{indent * depth}{self}")
        for feature_value, child in self:
            print(f"{indent * depth}{self.feature_name} = {feature_value}")
            child.print_subtree(depth + 1)
        
    def __str__(self):
        return f"id: {self.node_id}, rows: {self.rows_count}, feature: {self.feature_name}"
        
    def __iter__(self):
        return iter(self.children.items())


class BaselineSegmentationTree:

    ZERO_ENTROPY_DELTA = 0.0001

    def __init__(self, df, features, gm_column_name="gm", N=50):
        self._gm_column_name = gm_column_name
        self.N = N
        self._current_node_id = 0
        self.nodes_map = {}
        self.segments = {}
        self.root = self._build(features, {}, df)
        
    def predict(self, inputs):
        outputs = []
        for index, row in inputs.iterrows():
            node = self.root
            while not node.is_leaf:
                feature_value = row[node.feature_name]
                node = node.children[feature_value]
            outputs.append(self.segments[node.node_id])
        return outputs
        
    def _is_nan(self, value):
        return str(value) == "nan"
        
    def _create_node(self, feature, rows_count):
        next_node_id = self._current_node_id
        self._current_node_id += 1
        node = Node(next_node_id, feature, rows_count)
        self.nodes_map[next_node_id] = node
        return node
        
    def _create_segment(self, leaf, feature_values, df, zero_entropy):
        self.segments[leaf.node_id] = Segment(leaf, feature_values, self._determine_gm_cutoffs(df), zero_entropy)
        
    def _determine_gm_cutoffs(self, df):
        _, bins = pd.qcut(df[self._gm_column_name].sort_values(), q=[0, .2, .4, .6, .8, 1], retbins=True, duplicates="drop")
        return bins
        
    def _calculate_entropy(self, df, feature):
        total_rows_count = len(df)
        entropy = 0
        for feature_value in df[feature].dropna().unique():
            rows_count = len(df[df[feature] == feature_value])
            probability = rows_count / total_rows_count
            entropy -= probability * math.log(probability)
        return entropy
        
    def _build(self, features, feature_values, df):        
        if len(features) == 0 or len(df) <= self.N:
            leaf = self._create_node("", len(df))
            self._create_segment(leaf, feature_values, df, False)
            return leaf
        
        features_entropy = {}
        for feature in features:
            features_entropy[feature] = self._calculate_entropy(df, feature)
        
        max_entropy_feature = max(features_entropy, key=features_entropy.get)
        max_entropy = features_entropy[max_entropy_feature]
        
        if max_entropy > BaselineSegmentationTree.ZERO_ENTROPY_DELTA:
            current_node = self._create_node(max_entropy_feature, len(df))
            max_entropy_feature_values = df[max_entropy_feature].dropna().unique()
            for feature_value in max_entropy_feature_values:
                new_features = [f for f in features if f != max_entropy_feature]
                new_df = df[df[max_entropy_feature] == feature_value]
                new_feature_values = feature_values.copy()
                new_feature_values[current_node.feature_name] = feature_value
                child = self._build(new_features, new_feature_values, new_df)
                current_node.children[feature_value] = child
        else:
            current_node = self._create_node("", len(df))
            self._create_segment(current_node, feature_values, df, True)
            
        return current_node
        
        
class BaselineSegmentationTreeVisualizer:

    def __init__(self, tree, format="png",
                 internal_color="#BBBB00", leaf_color="#008800",
                 zero_entropy_color="#00CC00", problematic_color="#CC0000",
                 edge_label_color="#0000BB"):
        self.tree = tree
        self.format = format
        self.internal_color = internal_color
        self.leaf_color = leaf_color
        self.zero_entropy_color = zero_entropy_color
        self.problematic_color = problematic_color
        self.edge_label_color = edge_label_color
        self.dot = None
    
    def export(self, path):
        self.dot = Digraph(format=self.format)
        self._traverse(self.tree.root)
        self.dot.render(path, view=True)
        
    def _adapt_str(self, obj):
        return str(obj).replace(", ", "\n")
        
    def _create_node(self, node):
        if node.is_leaf:
            segment = self.tree.segments[node.node_id]
            obj = segment
            color = self.problematic_color if segment.is_problematic else \
                    (self.zero_entropy_color if segment.zero_entropy else self.leaf_color)
        else:
            obj = node
            color = self.internal_color
        self.dot.node(str(node.node_id), self._adapt_str(obj), style="filled", color=color)
        
    def _create_edge(self, from_id, to_id, label):
        self.dot.edge(str(from_id), str(to_id), label, fontcolor=self.edge_label_color)
        
    def _traverse(self, node):
        self._create_node(node)
        for feature_value, child in node:
            self._create_node(child)
            self._create_edge(node.node_id, child.node_id, f"{feature_value}")
            self._traverse(child)
