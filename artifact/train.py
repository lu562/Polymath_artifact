# MIT License

# Copyright (c) [2021] [Donghang Lu]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
from io import StringIO 
from IPython.display import Image  
import pydotplus
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_output(values):
    return np.argmax(values)

if __name__ == "__main__":

    file_name = "nursery.data"
    col_names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', "result"]
    dataset = pd.read_csv("nursery.data", header=None, names=col_names)
    feature_cols = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']

    # change the format of imput
    dataset.parents.replace("usual", 0 ,inplace = True)
    dataset.parents.replace("pretentious", 1 ,inplace = True)
    dataset.parents.replace("great_pret", 2 ,inplace = True)
    dataset.has_nurs.replace("proper", 0 ,inplace = True)
    dataset.has_nurs.replace("less_proper", 1 ,inplace = True)
    dataset.has_nurs.replace("improper", 2 ,inplace = True)
    dataset.has_nurs.replace("critical", 3 ,inplace = True)
    dataset.has_nurs.replace("very_crit", 4 ,inplace = True)
    dataset.form.replace("complete", 0 ,inplace = True)
    dataset.form.replace("completed", 1 ,inplace = True)
    dataset.form.replace("incomplete", 2 ,inplace = True)
    dataset.form.replace("foster", 3 ,inplace = True)
    dataset.children.replace("1", 0 ,inplace = True)
    dataset.children.replace("2", 1 ,inplace = True)
    dataset.children.replace("3", 2 ,inplace = True)
    dataset.children.replace("more", 3 ,inplace = True)
    dataset.housing.replace("convenient", 0 ,inplace = True)
    dataset.housing.replace("less_conv", 1 ,inplace = True)
    dataset.housing.replace("critical", 2 ,inplace = True)
    dataset.finance.replace("convenient", 0 ,inplace = True)
    dataset.finance.replace("inconv", 1 ,inplace = True)
    dataset.social.replace("nonprob", 0 ,inplace = True)
    dataset.social.replace("slightly_prob", 1 ,inplace = True)
    dataset.social.replace("problematic", 2 ,inplace = True)
    dataset.health.replace("recommended", 0 ,inplace = True)
    dataset.health.replace("priority", 1 ,inplace = True)
    dataset.health.replace("not_recom", 2 ,inplace = True)


    X = dataset[feature_cols]
    Y = dataset.result

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # Create Decision Tree classifer object
    max_depth = 11
    clf = DecisionTreeClassifier(max_depth)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    max_depth = clf.tree_.max_depth
    value = clf.tree_.value 

    parent = [-1 for _ in range(2 ** (max_depth + 1))]
    Is_left_child = np.zeros(shape=2 ** (max_depth + 1), dtype=np.int64)
    Is_right_child = np.zeros(shape=2 ** (max_depth + 1), dtype=np.int64)

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=2 ** (max_depth + 1), dtype=np.int64)
    is_leaves = np.zeros(shape=2 ** (max_depth + 1), dtype=bool)
    results = np.zeros(shape=2 ** (max_depth + 1), dtype=np.int64)
    poly = {}
    comparisons = {}
    id_index = n_nodes + 1
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            parent[children_left[node_id]] = node_id
            Is_left_child[children_left[node_id]] = 1
            parent[children_right[node_id]] = node_id
            Is_right_child[children_right[node_id]] = 1
        else:
            is_leaves[node_id] = True

    def complete_the_tree(parent_id, v, d, left_or_right):
        global id_index

        new_id = id_index
        id_index = id_index + 1
        node_depth[new_id] = d
        if left_or_right == 0:
            Is_left_child[new_id] = 1
        else:
            Is_right_child[new_id] = 1
        parent[new_id] = parent_id

        if d == max_depth:
            # it is a leaf node
            is_leaves[new_id] = True
            results[new_id] = v

            parents_left = []
            parents_right = []
            temp = new_id
            while parent[temp] != -1:
                if Is_left_child[temp] == 1:
                    parents_left.append(parent[temp])
                elif Is_right_child[temp] == 1:
                    parents_right.append(parent[temp])
                temp = parent[temp]
            poly[new_id] = [parents_left,parents_right]

        else:
            # expand the tree recursively
            comparisons[new_id] = [feature[0], threshold[0]]
            complete_the_tree(new_id, v, d + 1, 0)
            complete_the_tree(new_id, v, d + 1, 1)

    for i in range(n_nodes):
        if is_leaves[i]:
            if node_depth[i] != max_depth:
                comparisons[i] = [feature[0], threshold[0]]
                complete_the_tree(i, get_output(value[i]), node_depth[i] + 1, 0)
                complete_the_tree(i, get_output(value[i]), node_depth[i] + 1, 1)
            else:
                parents_left = []
                parents_right = []
                temp = i
                while parent[temp] != -1:
                    if Is_left_child[temp] == 1:
                        parents_left.append(parent[temp])
                    elif Is_right_child[temp] == 1:
                        parents_right.append(parent[temp])
                    temp = parent[temp]
                poly[i] = [parents_left,parents_right]
                results[i] = get_output(value[i])
        else:
            comparisons[i] = [feature[i], threshold[i]]

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    json_poly = json.dumps(poly, cls=NpEncoder)
    json_comparison = json.dumps(comparisons, cls=NpEncoder)
    json_value = json.dumps(results, cls=NpEncoder)
    with open('json_poly.json', 'w') as json_file:
        json_file.write(json_poly)
    with open('json_comparison.json', 'w') as json_file:
        json_file.write(json_comparison)
    with open('json_value.json', 'w') as json_file:
        json_file.write(json_value)


    # text_tree = export_text(clf, feature_names=feature_cols)
    # print(text_tree)

    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,  
    #                 filled=True, rounded=True,
    #                 special_characters=True,feature_names = feature_cols)

    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # graph.write_png('tree.png')
    # Image(graph.create_png())