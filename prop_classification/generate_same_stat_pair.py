import networkx as nx 
# from util import GraphProp
import multiprocessing 
from dataCollect_new import DataCollection_new


def test_pair_prob(something):
	# ana_tool = GraphProp()
	ana_tool = DataCollection_new()
	vertex_num = 100
	p=0.5
	counter = 0
	is_end = False
	while not is_end:
		# graph_list = []
		graphs_list_as_g6 = b""
		rep_dic = {}
		for index in range(2):

			graph = ana_tool.get_graph_by_gen('ER', vertex_num)
			# graph_list.append(graph)
			graphs_list_as_g6 += nx.to_graph6_bytes(graph)
			# stat = ana_tool.data_analysis(graph) 
			# stat = stat[:14]
			# stat.pop(1)
			stat = ana_tool.standardized_data_analysis(graph, vertex_num)
			stat_string = "".join(["{0:.2f} ".format(float(prop)) for prop in stat])

			if stat_string in rep_dic:
				rep_stat_list = rep_dic[stat_string]
				rep_stat_list.append(str(index))
				rep_dic[stat_string] = rep_stat_list
				if len(rep_stat_list) == 2:
					print(rep_stat_list)
					is_end = True
					break
			else:
				rep_dic[stat_string] = [str(index)]
			print(stat_string)

		counter += 1
	return str(counter), graphs_list_as_g6

# print(test_pair_prob(1))

def main():
	pool = multiprocessing.Pool()
	result = pool.map(test_pair_prob, range(10))
	name_or_path = './graphs_same_stat'+'_v='+str(100)
	stat, graphs_list_as_g6 = zip(*result)
	graphs_list_as_g6 = b''.join(list(graphs_list_as_g6))
	pool.close()
	with open(name_or_path + '.g6', 'wb') as graph_file, open(name_or_path + '.txt', 'w', newline='') as stat_file:
		stat_file.write(', '.join(stat))
		graph_file.write(graphs_list_as_g6)



if __name__ == '__main__':
	main()
