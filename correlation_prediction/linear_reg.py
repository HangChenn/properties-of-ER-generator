import numpy as np
import pandas as pd
import math
from itertools import combinations, combinations_with_replacement
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import ElasticNet

# from dataCollect_new import DataCollection_new
num_stat = 13

def load_data():
	dics = ['s_'+str(i) for i in range(1,11)]
	filenames = ['graphs_ER_100_'+str(x*100) for x in [1,2,4,8,16]]

	data= []

	for file_name in filenames:
		for dic in dics:
			matrix = np.loadtxt(dic+"/"+file_name+".csv", delimiter=",")
			data.extend(matrix)

	def remove_dumplicated(data):
		data = pd.DataFrame.from_records(data)
		data = data.round(5).drop_duplicates()
		return data.values

	data = remove_dumplicated(data)

	train_data = np.asarray(data[:int(0.8*len(data))])
	valid_data = np.asarray(data[int(0.8*len(data)):int(0.9*len(data))])
	test_data = np.asarray(data[int(0.9*len(data)):])
	return train_data, valid_data, test_data


def prepare_data(data_set, index):
	return prepare_data2(data_set, index, index)

def get_complement(index):
	index_cop = []
	for i in range(num_stat):
		if index.__class__ == index_cop.__class__ and i not in index:
			index_cop.append(i)
		elif index.__class__ == i.__class__ and i != index:
			index_cop.append(i)
	return index_cop

def prepare_data2(data_set, label, features_delete):
	y = data_set[:,label]
	x = np.delete(data_set, features_delete, 1)
	return x, y


def l1_distance(valid_y, predict_y, name):
	l1_dis_vec = np.absolute(predict_y-valid_y)
	print(name+"\tL1 Distance: \t", sum(l1_dis_vec)/len(valid_y), "\tand max dis: ",max(l1_dis_vec))
	return sum(l1_dis_vec)/len(valid_y)

def l2_distance(valid_y, predict_y, name):
	l2_dis_vec = np.square(predict_y-valid_y)
	print(name+"\tL2 Distance: \t", np.sqrt(sum(l2_dis_vec))/len(valid_y))


def linear_reg(train, valid, label_index, features, use_nonL_feature=True):
	train_x, train_y = prepare_data2(train,	label_index, get_complement(features))
	valid_x, valid_y = prepare_data2(valid,	label_index, get_complement(features))
	if use_nonL_feature:
		train_x, index = get_non_linear_feature(train_x)
		valid_x, index = get_non_linear_feature(valid_x)

		imp = SimpleImputer(missing_values=np.nan, strategy='mean')
		imp = imp.fit(train_x)

		train_x = imp.transform(train_x)
		valid_x = imp.transform(valid_x)
		print("non linear ", end='')
	# print("labels range\t", max(valid_y)-min(valid_y))
	# # print("labels std", np.std(valid_y))
	print("mean predictor L1\t",sum(np.absolute(valid_y-np.mean(train_y)))/len(valid_y))
	reg = LinearRegression().fit(train_x, train_y)
	if use_nonL_feature:
		print(reg.coef_)
	return l1_distance(valid_y, reg.predict(valid_x), "Linear")

	# l2_distance(valid_y, reg.predict(valid_x), "Linear")
	# dis = np.linalg.norm(reg.predict(valid_x)-valid_y)


def lasso(train, valid, label_index, features):
	train_x, train_y = prepare_data2(train,	label_index, get_complement(features))
	valid_x, valid_y = prepare_data2(valid,	label_index, get_complement(features))
	
	clf = Lasso(alpha=0.1**7,max_iter=10**6).fit(train_x, train_y)
	# clf = LinearRegression().fit(train_x, train_y)
	l1_distance(valid_y, clf.predict(valid_x), "Lasso")
	# l2_distance(valid_y, clf.predict(valid_x), "Lasso")
	# dis = np.linalg.norm(clf.predict(valid_x)-valid_y)
	# print("score for index ",index,"\t:",clf.score(valid_x, valid_y))
	# print("Lasso "+"\t\tDistance: ", dis)
	print(clf.coef_)
	print_index(clf.coef_, index)
	# print(clf.predict(valid_x))
	
	# print("Distance: ", dis)

def elastic_net(train, valid, label_index, features):
	train_x, train_y = prepare_data2(train,	label_index, get_complement(features))
	valid_x, valid_y = prepare_data2(valid,	label_index, get_complement(features))

	# print("labels range\t", max(valid_y)-min(valid_y))
	# # print("labels std", np.std(valid_y))
	# print("mean predictor L1\t",sum(np.absolute(valid_y-np.mean(train_y)))/len(valid_y))
	reg = ElasticNet().fit(train_x, train_y)
	return l1_distance(valid_y, reg.predict(valid_x), "elastic net")

	# l2_distance(valid_y, reg.predict(valid_x), "Linear")
	# dis = np.linalg.norm(reg.predict(valid_x)-valid_y)
def get_non_linear_feature(features):
	features = np.transpose(features)

	# remeber second order feature's name
	second_order_index = list(combinations_with_replacement(range(len(features)),2))
	# print(second_order_index)
	second_order_features = [np.multiply(ele1, ele2) for ele1, ele2 in combinations_with_replacement(features,2)]
	
	log_features = [np.log(ele) for ele in features]
	# log_features = np.nan_to_num(log_features)
	# log_features[log_features == np.-inf] = 1
	sqrt_features = [np.sqrt(ele) for ele in features]
	# sqrt_features = np.nan_to_num(sqrt_features)
	new_features = features
	new_features = np.append(new_features, second_order_features, axis=0)
	new_features = np.append(new_features, log_features, axis=0)
	new_features = np.append(new_features, sqrt_features, axis=0)
	new_features = np.transpose(new_features)

	
	new_index = [(ele, None) for ele in range(len(features))]
	new_index.extend(second_order_index)
	new_index.extend([(ele, 'log') for ele in range(len(features))])
	new_index.extend([(ele, 'sqrt') for ele in range(len(features))])


	return new_features, new_index
####################################################################################
#####  								BUG, NAME IS NOT CORRECT 					####
####################################33333333333333333333333333333333333333333333####
def print_index(coef_, index): 
	for i, num in zip(range(len(coef_)), coef_):
		if abs(num) > 0.1**5:
			tupe = index[i]
			if tupe[1] == None:
				print(stats_label[tupe[0]], " ", sep='',end='')
			elif tupe[1] == 'log':
				print("log(",stats_label[tupe[0]], ") ", sep='',end='')
			elif tupe[1] == 'sqrt':
				print("sqrt(",stats_label[tupe[0]], ") ", sep='',end='')
			else:
				print(stats_label[tupe[0]], "*", stats_label[tupe[1]], " ",sep='',end='')

	print()

def find_usaful_features():
	train, valid, test = load_data()
	def get_selected_props_list(num):
		props_list = []
		for i in range(1,num+1):
			props_list.extend(list(combinations(range(13), i)))
		return props_list
	props_list_list = get_selected_props_list(3)

	dfs = []
	for i in range(len(stats_label)):
		# print("****************************************************************")
		# print("target prop: ",stats_label[i])
		col_names =  ['features number', 'features', 'result', 'is useful']
		df = pd.DataFrame(columns=col_names)
		for props_list in props_list_list:
			if i in props_list:
				continue
			# print("features: ", *[stats_label[index] for index in props_list], sep='\t\t')
			dis = linear_reg(train, valid, i, list(props_list))
			df = df.append({'features number': len(props_list), 'features': set([stats_label[index] for index in props_list]),
			'result': dis}, ignore_index=True)
			# exit()
		dfs.append(df)


	effective_matrix = pd.DataFrame(columns=stats_label)
	for i in range(len(stats_label)):
		df = dfs[i]
		useful_features = [0] * len(stats_label)
		## only consider effect of adding one feature from two ele to three
		for index, row in df.loc[df['features number'] == 3].iterrows():
			with_ele_result = row['result']
			with_ele_features = row['features']
			for index2, row2 in df[(df['features number'] == 2) & (df['features'] <= with_ele_features)].iterrows():
				if (not math.isclose(row2['result'], 0,rel_tol=1e-16)) and (with_ele_result/row2['result'] < 4/5):
					feature_set = with_ele_features - row2['features']
					feature = feature_set.pop()
					useful_features[stats_label.index(feature)] += 1
		effective_matrix = effective_matrix.append(pd.Series(dict(zip(stats_label,useful_features)), name=stats_label[i]))
	print(effective_matrix)


def linear_and_lasso():
	train, valid, test = load_data()
	stats_label = ["GCC", "SCC", "APL", "r", "diam", "den", "Ce",
			"Cd", "Cc", "Cb", "Cei", "e_g_resist", "s_rad"]
	for i in range(len(stats_label)):  #[3,6]:
		# if stats_label[i] != 'Ce' and stats_label[i] != 'r':
		# 	continue
		print(stats_label[i])
		linear_reg(train, test, i, get_complement(i), use_nonL_feature=False)
		linear_reg(train, test, i, get_complement(i), use_nonL_feature=True)
		print()
		print("****************************")

if __name__ == '__main__':

	linear_and_lasso()

	# find_usaful_features()