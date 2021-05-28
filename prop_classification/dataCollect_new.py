import networkx as nx
import numpy as np
import math
import random


class DataCollection_new:
    def __init__(self,name=None, graphs=None):
        g = nx.star_graph(8)
        self.Freeman_degree_norm = self.freeman_centralization(nx.degree_centrality(g))
        self.Freeman_close_norm = self.freeman_centralization(nx.closeness_centrality(g))
        self.Freeman_between_norm = self.freeman_centralization(nx.betweenness_centrality(g))
        # need to change
        self.Freeman_eigen_norm = self.freeman_centralization(nx.eigenvector_centrality_numpy(g))


    def get_graph_by_list(self,index):
        return self.graphs[index]

    def get_graph_by_gen(self, type_of_graph, vertex_num):
        while True:
            if type_of_graph == "ER":
                graph = nx.fast_gnp_random_graph(vertex_num, 0.5)
            if type_of_graph == "ERlog":
                graph = nx.fast_gnp_random_graph(vertex_num, math.log(vertex_num)/vertex_num)
            if type_of_graph == "UN":
                graph = nx.fast_gnp_random_graph(vertex_num, random.random())
            if type_of_graph == "WS":
                ring_num = random.randint(2, vertex_num - 1)
                graph = nx.watts_strogatz_graph(vertex_num, ring_num, random.random())
            if type_of_graph == "BA":
                graph = nx.barabasi_albert_graph(vertex_num, random.randint(1, vertex_num - 1))
            if type_of_graph == "GE":
                graph = nx.random_geometric_graph(vertex_num, random.random())
            if nx.is_connected(graph):
                break
        return graph

    def standardized_data_analysis(self, graph, num_vertex):
        vec = self.data_analysis(graph)
        return vec
    
    def data_analysis(self, graph):
        data_vec = [0] * 13
        num_vertex = nx.number_of_nodes(graph)


        data_vec[0] = nx.average_clustering(graph)

        sq_values = list(nx.square_clustering(graph).values())
        data_vec[1] = sum(sq_values) / len(sq_values)

        g = nx.path_graph(num_vertex)
        data_vec[2] = nx.average_shortest_path_length(graph)/nx.average_shortest_path_length(g)

        data_vec[3] = nx.degree_pearson_correlation_coefficient(graph)
        if math.isnan(data_vec[3]) is True:
            data_vec[3] = 0

        data_vec[4] = nx.diameter(graph)/float((num_vertex-1))
        data_vec[5] = nx.density(graph)

        data_vec[6] = nx.edge_connectivity(graph)/float(num_vertex-1)

        g = nx.star_graph(num_vertex-1)
        Freeman_degree_norm = self.freeman_centralization(nx.degree_centrality(g))
        Freeman_close_norm = self.freeman_centralization(nx.closeness_centrality(g))
        Freeman_between_norm = self.freeman_centralization(nx.betweenness_centrality(g))
        # need to change
        Freeman_eigen_norm = self.freeman_centralization(nx.eigenvector_centrality_numpy(g))

        data_vec[7] = self.freeman_centralization(nx.degree_centrality(graph))/Freeman_degree_norm
        data_vec[8] = self.freeman_centralization(nx.closeness_centrality(graph))/Freeman_close_norm
        data_vec[9] = self.freeman_centralization(nx.betweenness_centrality(graph))/Freeman_between_norm


        # warning, the way it normalized may not correct
        data_vec[10] = self.freeman_centralization(nx.eigenvector_centrality_numpy(graph))/Freeman_eigen_norm

        egvl_lap = nx.laplacian_spectrum(graph)
        egvl_lap = np.sort(egvl_lap)
        egvl_lap = np.delete(egvl_lap,0,0)
        summ = 0
        for mu in egvl_lap:
            summ += (1/mu)

        summ = summ*num_vertex
        data_vec[11] = (num_vertex-1)/summ

        # for simple graph(adj matrix is symmetric), eigenvalue must be real number.
        egvl_adj = np.real(nx.adjacency_spectrum(graph))
        data_vec[12] = max(egvl_adj)/(num_vertex-1)

        return data_vec


    def freeman_centralization(self, list_centrality):
        max_cen = max(list_centrality.values())
        centralization = max_cen * len(list_centrality)
        for ele in list_centrality.values():
            centralization -= ele
        # print(centralization)
        return centralization

