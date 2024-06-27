import pandas as pd
import numpy as np

class GiniImpurityTree:
    def __init__(self, min_samples = 2, max_depth = 2):
        self.root = None

        self.min_samples = min_samples
        self.max_depth = max_depth

    def build_tree(self, data, depth=0):
        df = data.copy()

        x, y = df[:,:-1], df[:, -1]

        num_entries, num_features = np.shape(x)

        if (num_entries >= self.min_samples and depth <= self.max_depth):
            best_split = self.best_split(data, num_entries, num_features)
            if (best_split['info_gain'] > 0):
                left_subtree = self.build_tree(best_split['left_child'], depth + 1)
                right_subtree = self.build_tree(best_split['right_child'], depth + 1)

                return GiniNode(left_subtree, right_subtree, best_split['threshold'], best_split['info_gain'], best_split['feature_index'], 1)
        
        leaf_value = self.calc_leaf_value(y)
        return GiniNode(value=leaf_value)
            

    def best_split(self, data, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')

        
        for feature_index in range(num_features):
            specData = data[:, feature_index]
            possible_thresholds = np.unique(specData)
            for threshold in possible_thresholds:
                left_data, right_data = self.split(data, feature_index, threshold)
                if (len(left_data) > 0 and len(right_data) > 0):
                    y, left_y, right_y = data[:, -1], left_data[:, -1], right_data[:, -1]
                    info_gain = self.calc_info_gain(y, left_y, right_y)

                    if (info_gain > max_info_gain):
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left_child'] = left_data
                        best_split['right_child'] = right_data
                        best_split['info_gain'] = info_gain
                        max_info_gain = info_gain
                

        best_split['info_gain'] = max_info_gain
        return best_split

    def split(self, data, feature_index, threshold):
        left_branch = np.array([row for row in data if row[feature_index]<=threshold])
        right_branch = np.array([row for row in data if row[feature_index]>threshold])

        return left_branch, right_branch

    def calc_info_gain(self, parent, left, right):
        l_weight = len(left) / len(parent)
        r_weight = len(right) / len(parent)
        return self.gini_index(parent) - l_weight * self.gini_index(left) - r_weight * self.gini_index(right)

    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calc_leaf_value(self, y):
        y = list(y)
        # print(y)
        return y.count(1) / len(y)
    
    def train(self, X, Y):
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(data)

    def test(self, X, Y, threshold):
        l = []
        for i in range(len(X)):
            x = X[i]
            prob = self.make_prediction(x, self.root)
            prediction = prob >= threshold
            l.append(prediction)
            
            
        return l
    
    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_value  = x[tree.feature_index]
        if (feature_value <= tree.threshold):
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.weight)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

class GiniNode:
    def __init__(self, left=None, right=None, threshold=None, info_gain = None, feature_index = None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_index = feature_index
        
        self.value = value
