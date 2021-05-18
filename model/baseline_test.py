import numpy as np

from dataframe import *
from baseline import BaselineSegmentationTree, BaselineSegmentationTreeVisualizer

features_list = [manufacturing_region, product_family]
N = 250


def calculate_gm_distance(cuts1, cuts2):
    return np.sum((cuts1 - cuts2) ** 2) if cuts1.shape[0] == cuts2.shape[0] else -1


def format_segment(segment):
    feature_values = segment.feature_values
    rows = segment.leaf.rows_count
    gm_cutoffs = segment.gm_cutoffs

    feature_values_string = ', '.join(' = '.join((str(k), str(v))) for k, v in feature_values.items())
    feature_values_string = '[' + feature_values_string + ']'
    gm_cutoffs_string = ', '.join(f"{float(c):.3f}" for c in gm_cutoffs)
    gm_cutoffs_string = '[' + gm_cutoffs_string + ']'
    return f"{feature_values_string}, rows = {rows}, cuts = {gm_cutoffs_string}"


def print_distances(results):
    for result in results:
        segment_id1, segment_id2, distance = result
        segment1, segment2 = tree.segments[segment_id1], tree.segments[segment_id2]
        segment_string1, segment_string2 = format_segment(segment1), format_segment(segment2)
        print(segment_string1)
        print(segment_string2)
        print(f"distance = {distance:.5f}")
        print()


tree = BaselineSegmentationTree(df, features_list, N=N)

visualizer = BaselineSegmentationTreeVisualizer(tree)
visualizer.export(f"tree-{N}")

compatible = []
non_compatible = []
for i, (segment_id1, segment1) in enumerate(tree.segments.items()):
    for j, (segment_id2, segment2) in enumerate(tree.segments.items()):
        if j > i:
            distance = calculate_gm_distance(segment1.gm_cutoffs, segment2.gm_cutoffs)
            list_to_add = compatible if distance != -1 else non_compatible
            list_to_add.append((segment_id1, segment_id2, distance))

compatible.sort(key=lambda d: d[2], reverse=True)
print(f"Segment distances for N = {N}")
print(f"Features used: {features_list}")
print()

print_distances(compatible)
print_distances(non_compatible)
