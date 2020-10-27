#! /usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import tree

from decision_tree import build_tree

class RandomForest:
    def __init__(self, tgt_classes):
        self.tgt_classes = tgt_classes
        self.trees = []
    
    def add_tree(self, tree):
        self.trees.append(tree)
    
    def classify(self, input):
        tgt_num = {}

        for i, c in enumerate(self.tgt_classes):
            tgt_num[i] = 0
        
        for t in self.trees:
            out = t.classify(input)

            tgt_num[out] = tgt_num[out] + 1
        
        max, max_k = 0, -1
        for k in tgt_num:
            v = tgt_num[k]
            if v > max:
                max, max_k = v, k

        return max_k

    def set_debug(self, debug=True):
        for t in self.trees:
            t.set_debug(debug, propagate=True)
    
def build_forest(data, tree_num = 10, col_num=None, data_size=None, debug_forest=False, debug_tree=False):
    if(data_size == None):
        data_size = data.shape[0]

    if(col_num == None):
        col_num = data.columns.shape[0] - 1
    
    tgt_classes = data['target'].unique()
    col_names = data.drop('target', axis=1).columns

    forest = RandomForest(tgt_classes)

    for i in range(tree_num):
        if(debug_forest):
            print(" Tree number : %d"%(i+1))
            print("  Bagging data...")
        new_cols = []

        col_choice = np.random.choice(len(col_names), col_num, replace=False)

        for c in col_choice:
            new_cols.append(col_names[c])
        new_cols.append('target')
        
        ind_choice = np.random.choice(data.index, data_size, replace=True)

        bagged_data = data[new_cols].loc[ind_choice]

        tree_name = "tree%d"%(i+1)

        if(debug_forest):
            print("  Building tree '%s'"%tree_name)
        
        tree_root = build_tree(bagged_data, 5, root_name="tree_%d"%(i+1), debug=debug_tree, drop=False, max_tries=10, min_data=20)
        
        forest.add_tree(tree_root)

    return forest
