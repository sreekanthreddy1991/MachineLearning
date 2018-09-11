import glob

from data import Data
import math
import operator
import numpy as np

train_data = Data(fpath = "data_new/train.csv")
def calculate_info_gain(data, attr, label_attr, base_entropy):
    attr_possible_values = data.attributes[attr].possible_vals
    attr_entropy = 0.0
    totalEntries = float(data.__len__())
    if(totalEntries==0):
        return 1
    for val in attr_possible_values:
        attr_val_data = data.get_row_subset(attr, val)
        frac = len(attr_val_data)/totalEntries
        attr_entropy+= frac * calculate_entropy(attr_val_data, label_attr)
    return base_entropy - attr_entropy


def calculate_entropy(data, label_attr):
    label_values = data.get_column(label_attr)
    totalEntries = data.__len__()
    label_count_map = {}
    for val in label_values:
        if val in label_count_map.keys():
            label_count_map[val]+=1
        else:
            label_count_map[val] = 1
    entropy = 0.0
    for key in label_count_map:
        frac = float(label_count_map[key])/totalEntries
        entropy-= frac* math.log(frac, 2)
    return entropy

def get_best_attr(data, label_attr):
    best_attr = ""
    max_gain = 0.0
    base_entropy = calculate_entropy(data, label_attr)
    for attr in data.attributes.keys():
        info_gain = calculate_info_gain(data, attr, label_attr, base_entropy)
        if(info_gain > max_gain):
            max_gain = info_gain
            best_attr = attr
    if best_attr == "":
        print("empty best")
    return best_attr


def getMajorityLabel(label_values):
    label_map = {}
    for value in label_values:
        if value not in label_map.keys():
            label_map[value] = 1
        else:
            label_map[value]+=1
    max_label = ""
    max_count = 0
    for key in label_map.keys():
        if label_map[key] > max_count:
            max_count = label_map[key]
            max_label = key
    return max_label

def build_tree(s, label_attr):
    label_values = s.get_column(label_attr)
    unique_label_values = np.unique(label_values)
    major_label = getMajorityLabel(label_values)
    best_attr = ""
    if(len(unique_label_values)==1):
        return unique_label_values[0], 0
    else:
        if (len(s.attributes) == 0):
            best_attr = s.attributes.keys()[0]
        else:
            best_attr = get_best_attr(s, label_attr)
        best_attr_possible_vals = s.attributes[best_attr].possible_vals
        tree = {best_attr: {}}
        max_depth = 0
        for val in best_attr_possible_vals:
            data_subset = s.get_row_subset(best_attr, val)
            col_index = data_subset.get_column_index(best_attr)
            subset = np.delete(data_subset.raw_data, col_index, axis=1)
            attr_list = list(s.column_index_dict.keys())
            attr_list.remove(best_attr)
            attr_arr = np.asarray(attr_list)
            subset = np.insert(subset, 0, attr_arr, axis=0)
            new_data = Data(data=subset)
            tree[best_attr][val], depth = build_tree(new_data, label_attr)
            if depth > max_depth:
                max_depth = depth
    return tree, max_depth+1

def buildTreeWithDepth(s, label_attr, max_depth, current_depth=0):
    label_values = s.get_column(label_attr)
    unique_label_values = np.unique(label_values)
    major_label = getMajorityLabel(label_values)
    best_attr = ""
    if current_depth >= max_depth:
        return major_label
    if (len(unique_label_values) == 1):
        return unique_label_values[0]
    else:
        if(len(s.attributes)==0):
            best_attr = s.attributes.keys()[0]
        else:
            best_attr = get_best_attr(s, label_attr)
        best_attr_possible_vals = s.attributes[best_attr].possible_vals
        tree = {best_attr: {}}
        for val in best_attr_possible_vals:
            data_subset = s.get_row_subset(best_attr, val)
            col_index = data_subset.get_column_index(best_attr)
            subset = np.delete(data_subset.raw_data, col_index, axis=1)
            attr_list = list(s.column_index_dict.keys())
            attr_list.remove(best_attr)
            attr_arr = np.asarray(attr_list)
            subset = np.insert(subset, 0, attr_arr, axis=0)
            new_data = Data(data=subset)
            tree[best_attr][val] = buildTreeWithDepth(new_data, label_attr, max_depth, current_depth+1)
    return tree

def getLabelFromTree(tree, rowData, data):
    if (not isinstance(tree, dict)):
        return tree
    feature = list(tree.keys())[0]
    feature_index = data.column_index_dict[feature]
    feature_val = rowData[feature_index]
    subTree = tree[feature]
    if feature_val not in subTree:
        return "*"
    return getLabelFromTree(subTree[feature_val], rowData, data)


def get_accuracy(tree, data, label_attr):
    correct = 0
    label_index = data.column_index_dict[label_attr]
    for row in data.raw_data:
        label = getLabelFromTree(tree, row, data)
        if label == row[label_index]:
            correct+=1
    return float(correct)/data.__len__()

def cross_validation(dir_path, label_attr, depths):
    files = glob.glob(dir_path + '/*.csv')
    accuracy_map = {}
    std_map = {}
    for depth in depths:
        accuracy_list = []
        # print("Depth: "+str(depth))
        for i in range(len(files)):
            data0 = np.loadtxt(files[i], delimiter=",", dtype=str)
            data1 = np.loadtxt(files[(i + 1) % len(files)], delimiter=",", dtype=str, skiprows=1)
            data2 = np.loadtxt(files[(i + 2) % len(files)], delimiter=",", dtype=str, skiprows=1)
            data3 = np.loadtxt(files[(i + 3) % len(files)], delimiter=",", dtype=str, skiprows=1)
            data4 = np.loadtxt(files[(i + 4) % len(files)], delimiter=",", dtype=str)
            merged_data = np.concatenate((data0, data1, data2, data3), axis=0)
            training_data = Data(data=merged_data)
            testing_data = Data(data=data4)
            tree = buildTreeWithDepth(training_data, label_attr, depth, 0)
            # print("train accuracy: "+str(get_accuracy(tree, training_data, label_attr)))
            accuracy = get_accuracy(tree, testing_data, label_attr)
            # print("test accuracy: "+ str(accuracy))
            accuracy_list.append(accuracy)
        accuracy_map[depth] = np.mean(accuracy_list)
        std_map[depth] = np.std(accuracy_list)
    opt_depth = 0
    max_accuracy = 0
    for key in accuracy_map.keys():
        if accuracy_map[key] > max_accuracy:
            opt_depth = key
            max_accuracy = accuracy_map[key]
    return accuracy_map, std_map, opt_depth
print("========================================================")
print("a. Implementation of ID3 decision tree algorithm\n")
print("1. I have used python for implementation of ID3 algorithm and dictionary data structure to create the tree")
print("2. At any level if the Information Gain for all the features in the sublevel is same, then I chose the first feature in all of them")
print("3. In limiting depth, I found that accuracy is not 100% on training set itself, I believe it may be because of the anomolies in training set\n")

id3_tree, depth = build_tree(train_data, "label")
test_data = Data(fpath= "data_new/test.csv")
train_accuracy = get_accuracy(id3_tree, train_data, "label")
print("b. Error on train.csv: "+str((1-train_accuracy)*100)+"%\n")
test_accuracy = get_accuracy(id3_tree, test_data, "label")
print("c. Error on test.csv: "+str((1-train_accuracy)*100)+"%\n")
print("d. Maximum depth of the tree: "+str(depth)+"\n")
print("========================================================\n")
print("Limiting Depth:\n")
depths = [1, 2, 3, 4, 5, 10, 15]
accuracy_map, std_map, opt_depth = cross_validation("data_new/CVfolds_new", "label", depths)
print("a. Average cross validation accuracy and standard deviation for each depth are:\n")
print("Depth    | Average Accuracy(%)")
for key in accuracy_map.keys():
    print(str(key) + " | " + str(accuracy_map[key]*100))
print("\n")
print("Depth | Average Standard Deviation")
for key in std_map.keys():
    print(str(key)+" | "+str(std_map[key]))
print("\n")
print("maximum accuracy at depth: "+str(opt_depth))
print("\n")
print("Though we found the maximum accuracy at depth 10, there isn't much difference between accuracies for depths 5 and 10. "
      "We have seen from ID3 implementation as the maximum depth to be 6, so after this depth the accuracies will become saturated. "
      "So, I think it is better to select 5 as the optimum depth as there is no considerable differences between accuracies. "
      "Also choosing 5, we can improve the computation speed and lower depth trees with high accuracies are always preferable.\n")

tree_with_depth = buildTreeWithDepth(train_data, "label", 5, 0)
print("b. Test accuracy with depth: 5 is "+str(get_accuracy(tree_with_depth, test_data, "label")*100)+"%")
print("\n")
print("c. Observed accuracy without limiting depth is 1.0 and with depth of 5 is around 99.62%. "
      "I think limiting depth is a good idea, because having smaller trees is always better as traversing is faster than "
      "longer trees. If the test data is large, time taken to label all the data is lower with the smaller trees and that too with "
      "high accuracy. But we need to be careful while chosing the depth. We need to consider the balance between the "
      "depth we are going to chose and the performance we want to achieve with the chosen depth.")