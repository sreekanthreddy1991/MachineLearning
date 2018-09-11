from math import log
import glob
import sys
import operator
import itertools
import numpy as np

global_labels = ['len(fn) > len(ln)', 'has(middle name)', 'fn_first_last_char_eq', 'fn_before_ln', 'fn_second_letter_vowel', 'no_letters_ln_even', 'no_letters_fn_even', 'fn_ln_first_char_Same', '3_words_in_name', 'ln_first_last_char_eq']

unclassfied_label = '*'

def createDataSet(file_name_list, count_of_features = 6):
    
    file_contents = "";
    
    if isinstance(file_name_list, str):
        f = open(file_name_list, "r")
        file_contents = f.read()
    else:
        for file_name in file_name_list:
            f = open(file_name, "r")
            file_contents += f.read()
    
    file_data = file_contents.split("\n")
    
    data_without_features = []
    for row in file_data:
        row_list = row.split(" ", 1);
        if len(row_list) == 2:
            data_without_features.append(row.split(" ", 1))
    
    data_with_features = []
    
    #Calculating features
    for row in data_without_features:
        new_row = extractFeatures(row, count_of_features)
        data_with_features.append(new_row)
    
    return data_with_features, global_labels[:count_of_features]
    
def extractFeatures(dataRow, count_of_features = 6):
    label = dataRow[0]
    full_name = dataRow [1]
    
    names = full_name.lower().split(" ");
    
    #Is their first name longer than their last name?
    feature_1 = len(names[len(names) - 1]) < len(names[0])
    
    #Do they have a middle name?
    feature_2 = len(names) > 2
    
    #Does their first name start and end with the same letter? (ie "Ada")
    first_char_first_name = names[0][0]
    last_char_first_name = names[0][len(names[0]) - 1]
    feature_3 = first_char_first_name == last_char_first_name
    
    first_char_last_name = names[len(names) - 1][0]
    feature_4 = first_char_first_name < first_char_last_name
    
    #Is the second letter of their first name a vowel (a,e,i,o,u)?
    feature_5 = False
    if len(names[0]) != 1:
        second_char_first_name = names[0][1]
        feature_5 = second_char_first_name in ('a', 'e', 'i', 'o', 'u')
    
    
    #Is the number of letters in their last name even?
    feature_6 = len(names[len(names) - 1])%2 == 0
    
    #Is the number of letters in their first name even?
    feature_7 = len(names[0])%2 == 0
    
    #Do first name and last name start with the same letter?
    feature_8 = False
    
    if len(names) >= 2:
        last_name = names[len(names) - 1]
        first_char_last_name = last_name[0]
        feature_8 = first_char_first_name == first_char_last_name
        
    #Does the name have more than 3 words?
    feature_9 = len(names) > 3
    
    #4. Does their last name start and end with the same letter?
    feature_10 = False
    if len(names) == 1 :
        feature_10 = feature_3
    else:
        last_name = names[len(names) - 1]
        first_char_last_name = last_name[0]
        last_char_last_name = last_name[len(last_name) - 1]
        feature_10 = first_char_last_name == last_char_last_name
    
    feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]
    
    new_row = []
    new_row.extend(feature_list[:count_of_features])
    new_row.append(label);
    
    return new_row
    
def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    
    # the the number of unique elements and their occurance
    for row in dataSet:  
        currentLabel = row[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    entropy = 0.0
    
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        entropy -= prob * log(prob, 2)  # log base 2
    
    return entropy

def splitDataSet(dataSet, axis, value):
    #remove row from dataSet if row[axis] == value 
    returnDataSet = []
    
    for row in dataSet:    
        if row[axis] == value:
            
            # chop out axis used for splitting
            reducedFeatVec = row[:axis]  
            reducedFeatVec.extend(row[axis + 1:])
            
            returnDataSet.append(reducedFeatVec)
    return returnDataSet

def chooseBestFeatureToSplit(dataSet):
    # the last column is used for the labels
    numFeatures = len(dataSet[0]) - 1 
    
    baseEntropy = calcEntropy(dataSet)
    
    bestInfoGain = 0.0;
    bestFeature = -1
    
    infoGainArr = []
    
    # iterate over all the features
    for i in range(numFeatures):  
        
        # create a list of all the examples of this feature
        featList = [row[i] for row in dataSet]  
        
        # get a set of unique values
        uniqueVals = set(featList)  
        
        featureEntropy = 0.0
        
        for value in uniqueVals:
        
            subDataSet = splitDataSet(dataSet, i, value)
            
            prob = len(subDataSet) / float(len(dataSet))
            
            featureEntropy += prob * calcEntropy(subDataSet)

        # calculate the info gain; ie reduction in entropy
        infoGain = baseEntropy - featureEntropy  
        
        # compare this to the best gain so far
        if (infoGain >= bestInfoGain):
        
            # if better than current best, set to best
            bestInfoGain = infoGain  
            bestFeature = i
    
    # returns an integer
    return bestFeature  

def majorityCount(labelList):
    labelCount = {}
  
    #count occurrence of each label
    for label in labelList:
        if label not in labelCount.keys(): 
            labelCount[label] = 0
        labelCount[label] += 1
    
    #sort labels in reverse order of occurrence
    sortedClassCount = sorted(labelCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]

def createTree(dataSet, featureLabels):
    # extracting data
    
    #Get the label column from training data
    #-1 : Last column
    labelList = [row[-1] for row in dataSet]
    
    #if all labels are the same, stop splitting
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0], 1  
        
    # stop splitting when there are no more features in dataSet 
    if len(dataSet[0]) == 1:  
        return majorityCount(labelList), 1
    
    # use Information Gain
    
    #returns index of label
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = featureLabels[bestFeatureIndex]
    
    #build a tree recursively
    decisionTree = {bestFeatureLabel: {}}
    
    #delete chosen label : so that it is not considered again
    del (featureLabels[bestFeatureIndex])
    
    featureValues = [row[bestFeatureIndex] for row in dataSet]
    uniqueFeatureValues = set(featureValues)
        
    for value in uniqueFeatureValues:
        # copy all of labels, so trees don't mess up existing labels
        subLabels = featureLabels[:]  
        
        reducedDataSet = splitDataSet(dataSet, bestFeatureIndex, value)
        
        decisionTree[bestFeatureLabel][value], depth = createTree(reducedDataSet, subLabels)
        
    return decisionTree, depth+1

def createKFoldTree(dataSet, featureLabels, max_depth, current_depth=1):
    # extracting data
    
    #Get the label column from training data
    #-1 : Last column
    labelList = [row[-1] for row in dataSet]
    
    if current_depth >= max_depth:
        return majorityCount(labelList)
    
    #if all labels are the same, stop splitting
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]  
    
    # stop splitting when there are no more features in dataSet 
    if len(dataSet[0]) == 1:  
        return majorityCount(labelList)
    
    # use Information Gain
    
    #returns index of label
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = featureLabels[bestFeatureIndex]
    
    #build a tree recursively
    decisionTree = {bestFeatureLabel: {}}
    
    #delete chosen label : so that it is not considered again
    del (featureLabels[bestFeatureIndex])
    
    featureValues = [row[bestFeatureIndex] for row in dataSet]
    uniqueFeatureValues = set(featureValues)
        
    for value in uniqueFeatureValues:
        # copy all of labels, so trees don't mess up existing labels
        subLabels = featureLabels[:]  
        
        reducedDataSet = splitDataSet(dataSet, bestFeatureIndex, value)
        
        decisionTree[bestFeatureLabel][value] = createKFoldTree(reducedDataSet, subLabels, max_depth, current_depth+1)
        
    return decisionTree

def classify(inputTree, featLabels, testVec):
    
    if isinstance(inputTree, str):
        return inputTree[0]
    
    firstFeature = inputTree.keys()[0]
    
    subTree = inputTree[firstFeature]
    
    featureIndex = featLabels.index(firstFeature)
    
    key = testVec[featureIndex]
    
    if key not in subTree:
        return unclassfied_label
        
    featureValue = subTree[key]
    
    if isinstance(featureValue, dict):
        classLabel = classify(featureValue, featLabels, testVec)
    else:
        classLabel = featureValue
    return classLabel
    
def accuracy(tree, file_name, count_of_features = 6):
    f = open(file_name, "r")
    file_data = f.read().split("\n")
    
    data_without_features = []
    for row in file_data:
        row_list = row.split(" ", 1);
        if len(row_list) == 2:
            data_without_features.append(row.split(" ", 1))
    
    correct_prediction = 0;
    
    for row in data_without_features:
        row_with_features = extractFeatures(row, count_of_features)
        label_by_tree = classify(tree, global_labels[:count_of_features], row_with_features)
        if(label_by_tree == row[0]):
            correct_prediction += 1
    
    accuracy = float(correct_prediction) / len(data_without_features)
    return accuracy

def crossValidation(file_path, depths):
    
    #files = ["DataSet/Updated_CVSplits/updated_training00.txt", "DataSet/Updated_CVSplits/updated_training01.txt", "DataSet/Updated_CVSplits/updated_training02.txt", "DataSet/Updated_CVSplits/updated_training03.txt"]
    
    files = glob.glob(file_path + '\\*.txt')
    
    depth_accuracy = {}
    depth_std = {}
    depth_tree = {}
    
    for depth in depths:
        
        accuracies = []
        
        for i in range(len(files)):
            file_1 = files[i]
            file_2 = files[ (i+1) % len(files) ]
            file_3 = files[ (i+2) % len(files) ]
            file_4 = files[ (i+3) % len(files) ]
            
            file_list = [file_1, file_2, file_3]
            
            count_of_features = 10
            k_Fold_training_Data, k_Fold_labels = createDataSet(file_list, count_of_features)
            k_Fold_tree = createKFoldTree(k_Fold_training_Data, k_Fold_labels, depth)
            
            accuracyValue = accuracy(k_Fold_tree, file_4, count_of_features)
            accuracies.append(accuracyValue)
        
        depth_accuracy[depth] = np.mean(accuracies)
        depth_std[depth] = np.std(accuracies)
        
    sortedAccuracyDepth = sorted(depth_accuracy.iteritems(), reverse=True)
    
    return sortedAccuracyDepth[0][0], depth_accuracy, depth_std
                
print("============================")
print("Implementation")
print("==============\n")
print("a. Implementation of ID3 Algorithm\n")

print("Features used :")
print("---------------")
print("\n1. Is their first name longer than their last name?")
print("2. Do they have a middle name?")
print("3. Does their first name start and end with the same letter? (ie 'Ada')")
print("4. Does their first name come alphabetically before their last name? (ie 'Dan Klein' because 'd' comes before 'k')")
print("5. Is the second letter of their first name a vowel (a,e,i,o,u)?")
print("6. Is the number of letters in their last name even?")

print("\nI faced the following situations during implementation :")
print("------------------------------------------------------")
print("\n1. I used Python to code my implementation. I used the 'dictionary' data structure to create the decision tree.")
print("\n2. The information gain for all features for a subtree was the same and was zero. In this situation, I picked the last feature seen while choosing the best feature.")
print("\n3. For a particular feature, after traversing the tree, only one value i.e. only 'False' was seen. Hence during classification, for the above path, if Test Data has 'True', it had to be marked as unclassified.")
print("\n4. For the the Tree received after Training, the accuracy of the tree for the Training Data itself was not 100%. I had to verify the consistency of the training data. After verification, I observed that there multiple rows with same features, but they were classified differently.")

print("\nb. Four other features")
print("----------------------")
print("1. Is the number of letters in their first name even?")
print("2. Do first name and last name start with the same letter?")
print("3. Does the name have more than 3 words?")
print("4. Does their last name start and end with the same letter?")

training_file_name = "DataSet/updated_train.txt"
test_file_name = "DataSet/updated_test.txt"
training_Data, labels = createDataSet(training_file_name)
ID3_Tree, ID3_Depth = createTree(training_Data, labels)
#print("ID3 Tree : \n" + str(ID3tree) + "\n")
training_accuracy = accuracy(ID3_Tree, training_file_name)
testing_accuracy = accuracy(ID3_Tree, test_file_name)

print("\nc. Error on Training Data")
print("--------------------------")
error_train = (1-training_accuracy)*100;
print("Error : "+str(error_train)+"%");

print("\nd. Error on Test Data")
print("----------------------")
error_test = (1-testing_accuracy)*100;
print("Error : "+str(error_test)+"%");

print("\ne. Depth of Tree")
print("-----------------")
print("Depth of observed ID3 Tree : " + str(ID3_Depth))

print("\n\nLimiting Depth")
print("==============\n")

print("Features used :")
print("---------------")
print("\n1. Is their first name longer than their last name?")
print("2. Do they have a middle name?")
print("3. Does their first name start and end with the same letter? (ie 'Ada')")
print("4. Does their first name come alphabetically before their last name? (ie 'Dan Klein' because 'd' comes before 'k')")
print("5. Is the second letter of their first name a vowel (a,e,i,o,u)?")
print("6. Is the number of letters in their last name even?")
print("7. Is the number of letters in their first name even?")
print("8. Do first name and last name start with the same letter?")
print("9. Does the name have more than 3 words?")
print("10. Does their last name start and end with the same letter?")

print("\n\na. 4-fold Cross Validation")
print("-------------------------")

k_fold_file_path = "DataSet/Updated_CVSplits"
depth_variety = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
chose_depth, depth_accuracy, depth_std = crossValidation(k_fold_file_path, depth_variety)

print("Average Cross Validation Accuracy for each Depth : \n")
for key in depth_accuracy:
    print(str(key) + " : " + str(depth_accuracy[key] * 100) +"%")
    
print("\nStandard Deviation for each Depth : \n")
for key in depth_std:
    print(str(key) + " : " + str(depth_std[key]))
    
print("\nChosen Depth : " + str(chose_depth))
print("This depth has been chosen since it has the highest cross validation accuracy.")

print("\nb. Accuracy on Test Data")
print("-------------------------")

count_of_features = 10
training_Data, labels = createDataSet(training_file_name, count_of_features)
k_Fold_Tree = createKFoldTree(training_Data, labels, chose_depth)
k_fold_testing_accuracy = accuracy(k_Fold_Tree, test_file_name, count_of_features)

print("Accuracy : "+str(k_fold_testing_accuracy * 100)+"%");

print("\n\n============================")
