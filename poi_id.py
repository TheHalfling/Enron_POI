#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Added in a number of features to the original list of poi and salary
features_list = ['poi','salary', 'director_fees', 'total_stock_value',
                 'total_payments', 'exercised_stock_options',
                 'from_poi_to_this_person', 'deferral_payments'
                 ] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Remove the total line, only want those related to individual people or items
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### create new feature that is the ration of emails comparing those sent to
### those received.  Intention is to show a pattern of communication.
for x in my_dataset:
    if (my_dataset[x]['from_this_person_to_poi'] == 'NaN') or (my_dataset[x]
    ['from_poi_to_this_person'] == 'NaN'):
        my_dataset[x]['email_ratio'] = 0.0
    elif my_dataset[x]['from_poi_to_this_person'] != 0:
        my_dataset[x]['email_ratio'] = float(float(my_dataset[x]
        ['from_this_person_to_poi'])/
        float(my_dataset[x]['from_poi_to_this_person']))
    else:
        my_dataset[x]['email_ratio'] = 0.0
      

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Also tried those classifiers listed below, but did not acheive desired 
### results with them.

# clf = GaussianNB()
# clf = tree.DecisionTreeClassifier(max_depth = 10)
# clf = tree.DecisionTreeClassifier()
# clf = AdaBoostClassifier(n_estimators = 100)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.metrics import accuracy_score

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
 
clf.fit(features_train, labels_train)
 
 
t0 = time()
y_pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

y_true = labels_test

accuracy = accuracy_score(y_pred, labels_test)
print "Accuracy ", accuracy

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print "Precision ",precision_score(y_true, y_pred)
print "Score ",recall_score(y_true, y_pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)