from util import GraphProp

import math
import csv
import networkx as nx
import numpy as np
import random

from scipy import stats
import matplotlib.pyplot as plt

class generator:
	"""docstring for generator"""
	def __init__(self, model_name, p=None, cluster=1,name=None, block_list= None):
		self.model_name = model_name
		self.block_list = block_list
		self.params = p
		self.cluster = cluster
		self.ana_tool = GraphProp()
		self.name = name

	''' only work for simple graph(no self-loop, direct, weight) '''
	def calc_store_simple_graph(self, sample_size, vertex_num, add_path='', v_list=None):
		path = "./generated_data/"
		if add_path != '':
			path = path + add_path + '/'
		if self.name == None:
			p_str = ''
			if self.params != None:
				p_str = "_p="+ str(self.params)
			file_name = path+self.model_name+"_v="+str(vertex_num)+p_str
		else:
			file_name = path+self.name

		graphs_list_as_g6 = b""
		stat_list = []
		label_list = []
		with open(file_name + '.csv', 'a', newline='') as stat_file, open(file_name + '.g6', 'ab') as graph_file, open(file_name + '_label.csv', 'a', newline='') as label_file:
			stat_file_writer = csv.writer(stat_file, delimiter=',')
			label_file_writer = csv.writer(label_file, delimiter=',')
			for i in range(sample_size):
				if (i+1) % 10000 == 0:
					stat_file_writer.writerows(stat_list)
					label_file_writer.writerows(label_file)
					graph_file.write(graphs_list_as_g6)
					graphs_list_as_g6 = b""
					stat_list = []
					label_list = []

				p = self.params
				if self.model_name == "UN":
					p = np.random.rand()					
				graph = self.ana_tool.generate(self.model_name, vertex_num, p, self.cluster, v_list=v_list)

				node_label = [0]*graph.number_of_nodes()
				if self.cluster != 1:
					for label, partition in enumerate(graph.graph['partition']):
						for node_id in partition:
							node_label[node_id] = label

				label_list.append(node_label)
				graphs_list_as_g6 += nx.to_graph6_bytes(graph)
				stat = self.ana_tool.data_analysis(graph) 
				if(self.model_name == "UN"):
					stat.append(p)
				stat_list.append(stat)

			print(stat_list[0])
			stat_file_writer.writerows(stat_list)
			label_file_writer.writerows(label_list)
			graph_file.write(graphs_list_as_g6)


'''
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
'''


def get_normed_ramlist_gen(expected_p, weight_list, min_p=0, max_p=1):
	max_p = 1 if max_p > 1 else max_p
	min_p = 0 if min_p < 0 else min_p
	if expected_p < 0:
		return np.zeros(len(weight_list))
	not_qulified = True
	norm_list = []
	while not_qulified:	
		norm_list = np.random.rand(len(weight_list))
		norm_list = [x*(max_p-min_p)+min_p for x in norm_list]
		weigtht_sum = np.dot(norm_list,weight_list)
		norm_list = [p/weigtht_sum*sum(weight_list)*expected_p for p in norm_list]
		not_qulified = False
		for p in norm_list:
			if p >max_p or p < min_p:
				not_qulified = True
	return norm_list

def get_symm_h_p_cluster(n_list,e_p=0.5):
	# e_p=2*math.log(sum(n_list))/sum(n_list)
	e_p_in_cluster = 1.5*e_p


	diag_weight = np.asarray([n*(n-1)//2 for n in n_list])
	diag_p = get_normed_ramlist_gen(e_p_in_cluster, diag_weight,e_p,1.5*e_p_in_cluster)

	block_weight_between = []
	# based on nodes number, calculate the weight for each between cluster
	for i in range(1,len(n_list)):
		first_cluster_v_num = np.asarray(n_list[0:i])
		second_cluster_v_num = n_list[i]
		block_weight_between.extend(first_cluster_v_num*second_cluster_v_num)

	e_p_between =  (e_p*sum(n_list)*(sum(n_list)-1)//2- np.dot(diag_weight,diag_p))/sum(block_weight_between)
	print("expected p between blocks", e_p_between, sep=' ')
	r = get_normed_ramlist_gen(e_p_between, block_weight_between, 0, e_p)

	sym = np.zeros((len(n_list),len(n_list)))
	for i in range(len(n_list)):
		t = i*(i-1)//2
		sym[i,0:i] = r[t:t+i]
		sym[0:i,i] = r[t:t+i]
		sym[i,i] = diag_p[i]

	weight = np.zeros((len(n_list),len(n_list)))
	for i in range(len(n_list)):
		t = i*(i-1)//2
		weight[i,0:i] = block_weight_between[t:t+i]
		# weight[0:i,i] = block_weight_between[t:t+i]
		weight[i,i] = diag_weight[i]
	# print(weight)
	return sym



'''
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
'''
def setting1_vary():
	sample_size = 1000

	vertex_num = 100
	# generator_type_list = ["ER", "BA", "SBM"]
	generators = []

	expected_p = 0.5 
	def get_v_list(cluster,v_num):
		vertex_list = [math.ceil(v_num/cluster)]*(cluster-1)
		vertex_list.append(v_num-math.ceil(v_num/cluster)*(cluster-1))
		return vertex_list

	# generators.append(generator("ER",  p=expected_p, name="ER_v=100"))
	generators.append(generator("GE",  p=expected_p, name="GE_v=100"))
	# generators.append(generator("ER",  p=0, name="ER_v_vary"+'_p=logn'))
	# v_list = get_v_list(4,vertex_num)
	# generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list,expected_p), cluster=4, name="SBM_v=100"))
	generators.append(generator("BA", p=expected_p,name="BA_v=100"))
	generators.append(generator("WS", p=expected_p,name="WS_v=100"))
	
	
	for i in range(1,10):
		for gen in generators:
			print(gen.model_name, gen.params, sep=' ')
			gen.calc_store_simple_graph(sample_size, vertex_num, add_path='mix_generators/s'+str(i))
			print('done')



def SBM_diff_p_same_block():
	# np.random.seed(2020)

	sample_size = 1000
	instances_num = 20
	expected_p = 0.5
	# sample_size = 10
	# instances_num = 1

	vertex_num = 100
	add_path = 'SBM_V=100/num_p=20/p=0.5'
	# generator_type_list = ["ER", "BA", "SBM"]
	generators = []
	def get_v_list(cluster,v_num):
		vertex_list = [math.ceil(v_num/cluster)]*(cluster-1)
		vertex_list.append(v_num-math.ceil(v_num/cluster)*(cluster-1))
		return vertex_list
		# rand_v = np.random.rand(clusters_num)
		# rand_v = rand_v/sum(rand_v)*vertex_num
		# rand_v = np.rint(rand_v).astype(int)
		# rand_v[-1] = rand_v[-1] + vertex_num - sum(rand_v)
		# return rand_v
		

	for i in range(instances_num):	
		generators.append(generator("ER",  p=expected_p, name="SBM_v="+str(vertex_num)+'_c=1'))
		v_list = get_v_list(2,vertex_num)
		# print(v_list)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list,expected_p), cluster=2, block_list=v_list, name="SBM_v="+str(vertex_num)+'_c=2'))
		v_list = get_v_list(3,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list,expected_p), cluster=3, block_list=v_list, name="SBM_v="+str(vertex_num)+'_c=3'))
		v_list = get_v_list(4,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list,expected_p), cluster=4, block_list=v_list, name="SBM_v="+str(vertex_num)+'_c=4'))
		v_list = get_v_list(5,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list,expected_p), cluster=5, block_list=v_list, name="SBM_v="+str(vertex_num)+'_c=5'))
	


	# with open('./generated_data/SBM_h_p_same_block_logn/v_list_p_matrix.txt','w') as f:
	# with open('./generated_data/SBM_h_p_same_block/v_list_p_matrix.txt','w') as f:
	old_path = add_path
	for i in range(1,10):
		add_path = old_path+'/s'+str(i)
		with open('./generated_data/'+add_path+'/v_list_p_matrix.txt','w') as f:
			for gen in generators:
				if gen.cluster == 1:
					# gen = generator("ER",  p=2*math.log(vertex_num)/vertex_num, name="SBM_v="+str(vertex_num)+'_c=1')
					gen.calc_store_simple_graph(sample_size//instances_num, vertex_num, add_path=add_path)
				if gen.cluster > 1:
					v_list =gen.block_list
					f.write("block |V| list: ["+', '.join(str(v) for v in v_list)+']\n')
					[f.write('['+ ', '.join(str(p) for p in p_i)+']\n') for p_i in gen.params]
					f.write('\n')
					gen.calc_store_simple_graph(sample_size//instances_num, vertex_num, add_path=add_path, 
					v_list=v_list)
				print(add_path)
				print('done')



def SBM_diff_p_diff_block():
	np.random.seed(20200420)

	sample_size = 1000
	instances_num = 10

	vertex_num = 100
	# generator_type_list = ["ER", "BA", "SBM"]
	generators = []
	def get_v_list(remain_clusters,remain_v):
		if remain_clusters == 1:
			return [remain_v]
		min_v = 2
		max_v = min(remain_v/remain_clusters*2, 0.8*remain_v)
		# print(max_v)
		v = np.ceil(np.random.rand(1)[0] * max_v)
		v = int(v) if v > min_v else min_v
		return [v]+get_v_list(remain_clusters-1, remain_v-v)
		# rand_v = np.random.rand(clusters_num)
		# rand_v = rand_v/sum(rand_v)*vertex_num
		# rand_v = np.rint(rand_v).astype(int)
		# rand_v[-1] = rand_v[-1] + vertex_num - sum(rand_v)
		# return rand_v
		

	for i in range(instances_num):	
		generators.append(generator("ER",  p=0.5, name="SBM_v=100"+'_c=1'))
		v_list = get_v_list(2,vertex_num)
		# print(v_list)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list), cluster=2, block_list=v_list, name="SBM_v=100"+'_c=2'))
		v_list = get_v_list(3,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list), cluster=3, block_list=v_list, name="SBM_v=100"+'_c=3'))
		v_list = get_v_list(4,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list), cluster=4, block_list=v_list, name="SBM_v=100"+'_c=4'))
		v_list = get_v_list(5,vertex_num)
		generators.append(generator("SBM", p=get_symm_h_p_cluster(v_list), cluster=5, block_list=v_list, name="SBM_v=100"+'_c=5'))
	


	# with open('./generated_data/SBM_diff_p_diff_block/v_list_p_matrix.txt','w') as f:
	# 	for gen in generators:
	# 		if gen.cluster >1:
	# 			v_list =gen.block_list
	# 			f.write("block |V| list: ["+', '.join(str(v) for v in v_list)+']\n')
	# 			[f.write('['+ ', '.join(str(p) for p in p_i)+']\n') for p_i in gen.params]
	# 			f.write('\n')
	# 		gen.calc_store_simple_graph(sample_size//instances_num, vertex_num, add_path='SBM_diff_p_diff_block', 
	# 			v_list=v_list)
	# 		print('done')

	with open('./generated_data/SBM_difficulty_test/set3/v_list_p_matrix.txt','w') as f:
		for gen in generators:
			if gen.cluster == 1:
				gen = generator("ER",  p=2*math.log(vertex_num)/vertex_num, name="SBM_v="+str(vertex_num)+'_c=1')
				gen.calc_store_simple_graph(sample_size//instances_num, vertex_num, add_path='SBM_difficulty_test/set3')
			if gen.cluster > 1:
				v_list =gen.block_list
				f.write("block |V| list: ["+', '.join(str(v) for v in v_list)+']\n')
				[f.write('['+ ', '.join(str(p) for p in p_i)+']\n') for p_i in gen.params]
				f.write('\n')
				gen.calc_store_simple_graph(sample_size//instances_num, vertex_num, add_path='SBM_difficulty_test/set3', 
				v_list=v_list)
			print('done')



def test_den(model_name):

	ana_tool = GraphProp()
	not_stat = 0
	den_list = []
	for i in range(1000):
		graph = ana_tool.generate(model_name, 100)
		den = nx.density(graph)
		if den <0.45 or  den > 0.55:
			not_stat += 1
		den_list.append(den)
	print(model_name+" has "+str(not_stat)+ " disconnected per 1000 graphs")
	plt.hist(den_list,bins=50)
	plt.show()

	plt.hist([random.randint(0,28)-9 for i in range(1000)])
	plt.show()





if __name__ == '__main__':
	# SBM_diff_p_diff_block()
	# SBM_diff_p_same_block()
	setting1_vary()
	# test_den('ER')
	# test_den('BA')
	# test_den('GE')
	
	
	pass
