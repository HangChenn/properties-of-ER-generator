import networkx as nx
import math


def main():
	
	for vertex_num in range(200,2001, 100):
		print(vertex_num, ": ", end='')
		count = 0
		s_num = 1000
		for time in range(s_num):
			graph = nx.fast_gnp_random_graph(vertex_num, math.log(vertex_num)/vertex_num)
			if nx.is_connected(graph):
				count += 1
		print(count)


if __name__ == '__main__':
	main()