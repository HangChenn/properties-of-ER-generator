from util import load_data

from time import time
import numpy as np
import pandas as pd
import math

#import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.neural_network import MLPClassifier

from collections import Counter
from itertools import combinations, combinations_with_replacement


SELECTED_PROERTIES_NUMS_FOR_PREDICTION = 3

def main():
	# setup1(vary_v=False)
	# setup1(vary_v=True)
	# setup2(vary_v=False)
	# setup2(vary_v=True)
	# SBM_same_den()
	# notice_diff()
	# setting1_vary()
	# SBM_same_den_diff_p()
	SBM_same_den_diff_p_v()


def get_selected_props_list(len_prop, num):
	props_list = []
	for i in range(1,num+1):
		props_list.extend(list(combinations(range(len_prop), i)))
	return props_list


props_names = ["GCC", "SCC", "APL", "r", "diam", "den", "edge connectivity", "degree centralization", "closeness centralization",
    "betweeness centralization", "eigenvector centralization", "effective graph resistence", "spectral radius", "min degree"]
props_names = np.asarray(props_names)

def SBM_same_den_diff_p_v():

	path = './generated_data/final'
	name_list = []
	performance = []

	if path == "./generated_data/mix_generators":
		names=['ER', "SBM", "WS", "BA", "GE"]
		#  "WS", "BA", "GE"
		for name in names:
			name_list.append((name+"_v=100",name))
	elif path == './generated_data/final':
		names=["WS", "BA", "GE"]

		for name in names:
			name_list.append((name+"_v=100",name))
		name_list.append(("SBM_v=100_c="+str(1),'ER p=1/2'))
		for i in range(2,6):
			name_list.append(("SBM_v=100_c="+str(i),'SBM '+str(i)+' blocks'))
	else:
		for i in range(1,6):
			name_list.append(("SBM_v=100_c="+str(i),i))

	load_tool = load_data()
	X, y = load_tool.load_properties(path, name_list)
	n_samples, n_features = X.shape
	
	# you can remove few features by the following line
	# X_fea = np.column_stack((X[:,0], X[:,2:14]))
	props_list_list = get_selected_props_list(len(X_fea[0]), SELECTED_PROERTIES_NUMS_FOR_PREDICTION)
	data_result = []
	for props_list in props_list_list:
		X = X_fea[:,props_list]
		name = ", ".join(props_names[list(props_list)])
		result = [name]
		result.extend(use_machine_learning(X,y))
		data_result.append(result)
		print(result)
	dr_df = pd.DataFrame(data_result,columns=['name','Random Forest', 'Logistic Regression', 'SVM', 'NN','Random Forest std', 'Logistic Regression std', 'SVM std', 'NN std'])
	# dr_df = pd.DataFrame(data_result,columns=['name','Random Forest', 'Logistic Regression', 'SVM', 'NN'])
	dr_df.to_csv(path+'/performance.csv')
	# exit()



def modify_features(X, task):
	if task == 5:
		# use all features
		X_fea = X
	if task == 1:
		# only density
		# X_fea = np.column_stack((X[:,0], X[:,0]))
		X_fea = np.column_stack((X[:,5], X[:,5]))
	if task == 2:
		# only spectral radius and effective graph resistence
		X_fea = np.column_stack((X[:,10], X[:,11]))
	if task == 3:
		# use only degree sequencess
		X_fea = X[:,14:]
	if task == 4:
		# use 15 features
		X_fea = X[:,:15]
	return X_fea

def use_machine_learning(X,y, task=5):
	only_svm = False
	performance = []
	std = []
	X_fea = modify_features(X, task)
	train_X, valid_X, test_X = prepare_data(X_fea)
	train_y, valid_y, test_y = prepare_data(y)
	test_X = np.concatenate((valid_X, test_X), axis=0)
	test_y = np.concatenate((valid_y, test_y), axis=0)

	unique, counts = np.unique(test_y, return_counts=True)
	print(dict(zip(unique, counts)))

	cv = ShuffleSplit(n_splits=10, test_size=0.2)

	# X_den = test_X[:,6]
	# den_0, den_1 = {}, {}
	# den_0_fe, den_1_fe = [], []
	# for den, label in zip(X_den, test_y):
	# 	if label == 0:
	# 		den_0_fe.append(den)
	# 	if label == 1:
	# 		den_1_fe.append(den)
	# print('naive method that flip a fair coin when density is same: ', end='')
	# print(1-len(list((Counter(den_0_fe) & Counter(den_1_fe)).elements()))/len(test_y))
	

	X = X_fea
	# ----------------------------------------------------------------------
	# RandomForestClassifier of the generators dataset
	clf = RandomForestClassifier()
	scores = cross_val_score(clf, X, y, cv=cv)
	performance.append(scores.mean())
	std.append(scores.std())
	# clf.fit(train_X, train_y)
	# predict_y = clf.predict(test_X)
	# print("Random Forest accuracy score: %.3f"%accuracy_score(predict_y, test_y))
	# performance.append(accuracy_score(predict_y, test_y))
	# print("",end='') if only_svm else print(sklearn.metrics.confusion_matrix(test_y, predict_y))

	# ----------------------------------------------------------------------
	# Logistic Regression of the generators dataset
	clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	scores = cross_val_score(clf, X, y, cv=cv)
	performance.append(scores.mean())
	std.append(scores.std())
	# clf.fit(train_X, train_y)
	# predict_y = clf.predict(test_X)
	# print("Logistic Regression accuracy score: %.3f"%accuracy_score(predict_y, test_y))
	# performance.append(accuracy_score(predict_y, test_y))
	# print("",end='') if only_svm else print(sklearn.metrics.confusion_matrix(test_y, predict_y))
	# ----------------------------------------------------------------------
	# SVM of the generators dataset
	clf = SVC(gamma='scale')
	scores = cross_val_score(clf, X, y, cv=cv)
	performance.append(scores.mean())
	std.append(scores.std())
	# clf.fit(train_X, train_y)
	# predict_y = clf.predict(test_X)
	# print("SVM accuracy score: %.3f"%accuracy_score(predict_y, test_y))
	# performance.append(accuracy_score(predict_y, test_y))
	# print(sklearn.metrics.confusion_matrix(test_y, predict_y))
	# ----------------------------------------------------------------------
	'''https://scikit-learn.org/stable/auto_examples/neural_networks/
	plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py'''
	clf = MLPClassifier(solver='lbfgs', alpha=0.02,
			hidden_layer_sizes=[100, 100], random_state=1)
	scores = cross_val_score(clf, X, y, cv=cv)
	performance.append(scores.mean())
	std.append(scores.std())
	# clf.fit(train_X, train_y)
	# predict_y = clf.predict(test_X)
	# print("nn accuracy score: %.3f"%accuracy_score(predict_y, test_y))
	# performance.append(accuracy_score(predict_y, test_y))
	# print("",end='') if only_svm else print(sklearn.metrics.confusion_matrix(test_y, predict_y))

	performance.extend(std)
	return performance




def prepare_data(data):
	train_data = np.asarray(data[:int(0.8*len(data))])
	valid_data = np.asarray(data[int(0.8*len(data)):int(0.9*len(data))]) 
	test_data = np.asarray(data[int(0.9*len(data)):])

	return train_data, valid_data, test_data


if __name__ == '__main__':
	main()