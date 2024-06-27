import numpy as np

class WeightedTree:
    def __init__(self, X, Y, max_depth=2, min_samples=2):
        self.root = None
        self.X = X
        self.Y = Y

        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def build_tree(self, data, feature_index=0):
        df = data.copy()

        X,Y = df[:, :-1], df[:, -1]

        num_entries, num_features = np.shape(X)
        if (feature_index <= num_features):
            branches = self.create_branches(data, feature_index)

            for branch in branches.keys():
                branches[branch] = self.build_tree(branches[branch], feature_index + 1)
            
            total, leaf_val = self.calc_leaf_value(Y)
            return Node(weight=1, branch=branches, total = total, value = leaf_val)
        
        total, leaf_value = self.calc_leaf_value(Y)
        return Node(value = leaf_value, total = total, final = True)

    def create_branches(self, data, feature_index):
        specData = data[:, feature_index]
        features = np.unique(specData)
        branches = {}
        
        for feature in features:
            branches[feature] = np.array([row for row in data if row[feature_index]==feature])
        
        return branches
    
    def calc_leaf_value(self, Y):
        Y = list(Y)
        return len(Y), Y.count(1) / len(Y)
    
    def make_prediction(self, x, tree, feature_index=0):
        if tree.final: return tree.value
        feature_value = x[feature_index]
        try:
            return self.make_prediction(x, tree.branches[feature_value], feature_index + 1)
        except:
            # print('a')
            return tree.value
    
    def make_prediction_with_weights(self, x, tree, feature_index=0):
        if tree.final:
            return tree, tree.value * tree.weight
        
        feature_value = x[feature_index]
        try:
            return self.make_prediction_with_weights(x, tree.branches[feature_value], feature_index + 1)
        except:
            return tree, tree.value * tree.weight
        
    def weight_adjustment(self, X, Y, threshold=0.5):
        for i in range(len(X)):
            x = X[i]
            prob = self.make_prediction(x, self.root)
            prediction = prob >= threshold

            dataset = np.concatenate((X,Y), axis = 1)
            rows = list([row for row in dataset if (row[:-1] == x).all()])
            
            num = 0
            for ele in rows:
                if ele[-1] == prediction:
                    num += 1
            total = len(rows)
            if (prediction == Y[i]):
                if (prediction):
                    self.adjustWeight(x, self.root, 1.2)
                # else: 
                #     self.adjustWeight(x, self.root, -2)
            else:
                if (prediction):
                    self.adjustWeight(x, self.root, -1.2)
                # else:
                #     self.adjustWeight(x, self.root, 1.2)
            # else:
            #     if (prediction):
            #         self.adjustWeight(x, self.root, - 0.2)
            #     else:
            #         self.adjustWeight(x, self.root, 0.2)
    
    def adjustWeight(self, x, tree, adjust, feature_index=0):
        feature_value = x[feature_index]
        if tree.final: return
        # print(tree.weight)
        if (tree != self.root):
            tree.weight += adjust
        
        try:
            self.adjustWeight(x, tree.branches[feature_value], adjust, feature_index + 1)
        except:
            return
        
    def train(self, X, Y):
        data = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(data)

    def test(self, X, Y, threshold=0.5):
        l = []
        for i in range(len(X)):
            x = X[i]
            tree, prob = self.make_prediction_with_weights(x, self.root)
            prediction = prob >= threshold
            
            # if (Y[i]):
            #     tree.value = (tree.value * tree.total + 1) / (tree.total + 1)
            # else:
            #     tree.value = (tree.value * tree.total) / (tree.total + 1)
            # tree.value =
                
            if (prediction == Y[i]):
                if (prediction):
                    self.adjustWeight(x, self.root, 1 / tree.total)
                else:
                    self.adjustWeight(x, self.root, -1 / tree.total)
            else:
                if (prediction):
                    if (tree.value == 1):
                        tree.value = (tree.total) / (tree.total + 1)
                    self.adjustWeight(x, self.root, -1 / tree.total)
                else:
                    if (tree.value == 0):
                        tree.value = 0.2
                    self.adjustWeight(x, self.root, 2)
    
    def test2(self, X, Y, threshold=0.5):
        l = []
        for i in range(len(X)):
            x = X[i]
            tree, prob = self.make_prediction_with_weights(x, self.root)
            prediction = prob >= threshold
            

            # print(prediction)
            # print(Y[i])
            l.append(prediction)
            
        return l

class Node:
    def __init__(self, weight = 1, branch = None, total = None, value = None, final = False):
        self.weight = weight
        self.branches = branch

        self.final = final
        self.total = total
        self.value = value

    def create(self):
        for key in set(self.data[self.data.columns[0]]):
            specData = self.data[self.data[self.data.columns[0]] == key]
            if (len(specData) > 0):
                temp = Node(specData.drop(columns=specData.columns[0]), len(specData[specData[specData.columns[-1]] == 1]), len(specData))

                if (len(specData.columns) > 2):
                    temp.create()
                self.nodes[key] = temp
