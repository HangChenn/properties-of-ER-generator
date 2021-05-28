import networkx as nx
import dgl
import numpy as np
import math
import random
import torch as th
import pandas as pd

import matplotlib.pyplot as plt


class GraphProp:
    def __init__(self):
        pass

    def generate(self, type_of_graph, vertex_num, p=0.5, cluster=1,diff_block_size=False,v_list=None):
        if type_of_graph == "ER":
            graph = nx.fast_gnp_random_graph(vertex_num, p)
        if type_of_graph == "BA":
            # graph = nx.barabasi_albert_graph(vertex_num, random.randint(1, vertex_num - 1))
            var_num = random.randint(0,15)-5
            attatched_num = math.ceil((vertex_num-1)*p)-var_num
            graph = nx.barabasi_albert_graph(vertex_num, attatched_num)
        if type_of_graph == "SBM":
            vertex_list = [math.ceil(vertex_num/cluster)]*(cluster-1)
            vertex_list.append(vertex_num-math.ceil(vertex_num/cluster)*(cluster-1))
            if diff_block_size:
                rand_v = np.random.rand(cluster)
                rand_v = rand_v/sum(rand_v)*vertex_num
                rand_v = np.rint(rand_v).astype(int)
                rand_v[-1] = rand_v[-1] + vertex_num - sum(rand_v)
            if v_list is not None:
                vertex_list = v_list
            graph = nx.stochastic_block_model(vertex_list, p)
        if type_of_graph == "GE":
            while True:
                graph = nx.random_geometric_graph(vertex_num, p)
                den = nx.density(graph)
                if den >= (p - 0.03) and den <= (p + 0.03):
                    # print(den)
                    break

        if type_of_graph == "UN":
            graph = nx.fast_gnp_random_graph(vertex_num, p)
        if type_of_graph == "WS":
            # ring_num = random.randint(2, vertex_num - 1)
            ring_num = math.ceil((vertex_num-1)*(p-0.15))
            while True:
                graph = nx.newman_watts_strogatz_graph(vertex_num, ring_num, 0.47)
                den = nx.density(graph)
                if den >= (p - 0.03) and den <= (p + 0.03) and nx.is_connected(graph):
                    break

        return graph

    def connected_graph_generat(self, type_of_graph, vertex_num, p=0.5):
         while True:
            graph = generators(type_of_graph, vertex_num, p)
            if nx.is_connected(graph):
                return graph



    ''' 
    In the following section we evaluate properties include
    GCC, ACC, SCC, APL, r, diam, den, edge connectivity, degree centralization, closeness centralization (10)
    betweeness centralization, eigenvector centralization, effective graph resistence, spectral radius, min degree (5)
    other degree
    '''
    def data_analysis(self, graph):
        data_vec = []

        num_vertex = nx.number_of_nodes(graph)

        # GCC
        data_vec.append(nx.transitivity(graph))
        ### ACC
        data_vec.append(nx.average_clustering(graph))

        ### SCC
        sq_values = list(nx.square_clustering(graph).values())
        data_vec.append(sum(sq_values) / len(sq_values))

        ### APL
        try:
            g = nx.path_graph(num_vertex)
            data_vec.append(nx.average_shortest_path_length(graph)/nx.average_shortest_path_length(g))
        except Exception as e:
            # data_vec.append(np.nan)
            # data_vec.append(1.5)
            max_apl = 0
            for S in [graph.subgraph(c).copy() for c in nx.connected_components(graph)]:
                apl = nx.average_shortest_path_length(S)/nx.average_shortest_path_length(g)
                if apl >= max_apl:
                    max_apl = apl
            data_vec.append(max_apl)

        ### r
        try: 
            data_vec.append(nx.degree_assortativity_coefficient(graph))
        except Exception as e:
            if graph.number_of_edges() == 0:
                data_vec.append(0)
            else:
                print("Unexpected error")
                raise

        ### diam
        try:
            data_vec.append(nx.diameter(graph)/float(num_vertex-1))
        except Exception as e:
            # data_vec.append(np.nan)
            max_diam = 0
            for S in [graph.subgraph(c).copy() for c in nx.connected_components(graph)]:
                diam = nx.diameter(S)/float(num_vertex-1)
                if diam >= max_diam:
                    max_diam = diam
            data_vec.append(max_diam)
        

        ### den
        data_vec.append(nx.density(graph))

        ### edge connectivity
        data_vec.append(nx.edge_connectivity(graph)/float(num_vertex-1))

        ### degree centralization, closeness centralization, betweeness centralization, eigenvector centralization
        g = nx.star_graph(num_vertex-1)
        Freeman_degree_norm = self.freeman_centralization(nx.degree_centrality(g))
        Freeman_close_norm = self.freeman_centralization(nx.closeness_centrality(g))
        Freeman_between_norm = self.freeman_centralization(nx.betweenness_centrality(g))
        Freeman_eigen_norm = self.freeman_centralization(nx.eigenvector_centrality_numpy(g))
        data_vec.append(self.freeman_centralization(nx.degree_centrality(graph))/Freeman_degree_norm)
        data_vec.append(self.freeman_centralization(nx.closeness_centrality(graph))/Freeman_close_norm)
        data_vec.append(self.freeman_centralization(nx.betweenness_centrality(graph))/Freeman_between_norm)
        # warning, the way it normalized may not correct
        data_vec.append(self.freeman_centralization(nx.eigenvector_centrality_numpy(graph))/Freeman_eigen_norm)


        ### effective graph resistence 
        egvl_lap = nx.laplacian_spectrum(graph)
        egvl_lap = np.sort(egvl_lap)
        egvl_lap = np.delete(egvl_lap,0,0)
        summ = 0
        for mu in egvl_lap:
            summ += (1/mu)
        summ = summ*num_vertex
        data_vec.append((num_vertex-1)/summ)

        ### spectral radius
        # for simple graph(adj matrix is symmetric), eigenvalue must be real number.
        egvl_adj = np.real(nx.adjacency_spectrum(graph))
        data_vec.append(max(egvl_adj)/(num_vertex-1))

        ### degree sequence(small to large)
        degree_sequence = sorted([float(d/num_vertex) for n, d in graph.degree()], reverse=False)
        data_vec.extend(degree_sequence)

        return data_vec


    def freeman_centralization(self, list_centrality):
        max_cen = max(list_centrality.values())
        centralization = max_cen * len(list_centrality)
        for ele in list_centrality.values():
            centralization -= ele
        return centralization
        

class load_data():

    # return a list of 2-tuples, (feature, label) 
    def load_properties(self, file_path, filenames_type):
        np.random.seed(2020)
        data= []
        y=[]
        for file_name, label in filenames_type:
            # print(file_name)
            # matrix = np.loadtxt(file_path+"/"+file_name+".csv", delimiter=",")
            matrix = pd.read_csv(file_path+"/"+file_name+".csv", header=None, sep='\n')
            matrix = matrix[0].str.split(',', expand=True)
            matrix = matrix.fillna(0)

            matrix = np.asarray(matrix)

            # matrix = [ f[:14] for f in matrix]
            # print(matrix[1])
            matrix = zip(matrix,[label]*len(matrix))
            data.extend(matrix)
  
            # print(data[len(data)-1], data[len(data)-1][0])
        np.random.shuffle(data)
        # print(data)
        X,y = map(list, zip(*data))
        # print(X[0])
        # print(y[0])
        return np.array(X, dtype=float),np.array(y)


    def load_graphs(self, file_path, filenames_type):
        np.random.seed(2020)
        data = []
        y = []
        def prepare_data(data):
            train_data = data[:int(0.8*len(data))]
            valid_data = data[int(0.8*len(data)):int(0.9*len(data))]
            test_data = data[int(0.9*len(data)):]
            return train_data, valid_data, test_data 

        for file_name, label in filenames_type:
            g_list = nx.read_graph6(file_path+"/"+file_name+".g6")
            g_list = zip(g_list,[label]*len(g_list))

            data.extend(g_list)
        np.random.shuffle(data)

        graphs,y = map(list, zip(*data))
        train_set_X, valid_set_X, test_set_X = prepare_data(graphs)
        train_set_y, valid_set_y, test_set_y = prepare_data(y)
        train_set = my_graph_Dataset(train_set_X, train_set_y)
        valid_set = my_graph_Dataset(valid_set_X, valid_set_y)
        test_set = my_graph_Dataset(test_set_X, test_set_y)


        # return graphs, y
        return train_set, valid_set, test_set

    def load_graphs_with_props(self, file_path, filenames_type):
        np.random.seed(2020)
        data = []
        y = []
        def prepare_data(data):
            train_data = data[:int(0.8*len(data))]
            valid_data = data[int(0.8*len(data)):int(0.9*len(data))]
            test_data = data[int(0.9*len(data)):]
            return train_data, valid_data, test_data 


        for file_name, label in filenames_type:
            g_list = nx.read_graph6(file_path+"/"+file_name+".g6")
            p_list = np.loadtxt(file_path+"/"+file_name+".csv", delimiter=",")
            g_list = zip(zip(g_list,p_list),[label]*len(g_list))


            data.extend(g_list)
        np.random.shuffle(data)

        graphs,y = map(list, zip(*data))
        train_set_X, valid_set_X, test_set_X = prepare_data(graphs)
        train_set_y, valid_set_y, test_set_y = prepare_data(y)
        train_set = my_graph_Dataset2(train_set_X, train_set_y)
        valid_set = my_graph_Dataset2(valid_set_X, valid_set_y)
        test_set = my_graph_Dataset2(test_set_X, test_set_y)


        # return graphs, y
        return train_set, valid_set, test_set

    def load_graphs_with_props_node(self, file_path, filenames_type):
        np.random.seed(2020)
        data = []
        y = []
        def prepare_data(data):
            train_data = data[:int(0.8*len(data))]
            valid_data = data[int(0.8*len(data)):int(0.9*len(data))]
            test_data = data[int(0.9*len(data)):]
            return train_data, valid_data, test_data 


        for file_name, label in filenames_type:
            g_list = nx.read_graph6(file_path+"/"+file_name+".g6")
            print(file_name)
            p_list = np.loadtxt(file_path+"/"+file_name+".csv", delimiter=",")
            l_list = np.loadtxt(file_path+"/"+file_name+"_label.csv", delimiter=",")
            g_list = zip(zip(g_list,p_list),l_list)


            data.extend(g_list)
        np.random.shuffle(data)

        graphs,y = map(list, zip(*data))
        train_set_X, valid_set_X, test_set_X = prepare_data(graphs)
        train_set_y, valid_set_y, test_set_y = prepare_data(y)
        train_set = my_graph_Dataset_node_label(train_set_X, train_set_y)
        valid_set = my_graph_Dataset_node_label(valid_set_X, valid_set_y)
        test_set = my_graph_Dataset_node_label(test_set_X, test_set_y)


        # return graphs, y
        return train_set, valid_set, test_set


class my_graph_Dataset(object):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

        # preprocess
        for i in range(len(self.graphs)):
            self.graphs[i] = dgl.DGLGraph(self.graphs[i])
            # # add self edges
            # nodes = self.graphs[i].nodes()
            # self.graphs[i].add_edges(nodes, nodes)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class my_graph_Dataset2(object):
    def __init__(self, data, labels):
        graphs, properties = map(list, zip(*data))
        self.graphs = graphs
        self.labels = labels
        # self.properties = np.asarray(properties)

        self.properties = np.asarray([list(p[:8])+list(p[12:14]) for p in properties])
        # self.properties = np.column_stack((self.properties[:,12], self.properties[:,13]))
        

        # # preprocess
        for i in range(len(self.graphs)):
            temp_graph = dgl.DGLGraph()
            temp_graph.from_networkx(self.graphs[i])
            node_ids = list(range(len(self.graphs[i].nodes)))
            np.random.shuffle(node_ids)

            # store random ID for each graph
            ndata = th.zeros((len(node_ids),1))
            for n_id, j in zip(node_ids, range(len(node_ids))):
                ndata[j] = n_id
            temp_graph.ndata['s'] = ndata
        #     nx.set_node_attributes(self.graphs[i], nx.betweenness_centrality(self.graphs[i]), 'b')
        #     nx.set_node_attributes(self.graphs[i], nx.closeness_centrality(self.graphs[i]), 'c')
        #     nx.set_node_attributes(self.graphs[i], nx.eigenvector_centrality_numpy(self.graphs[i]), 'e')
        #     temp_graph.from_networkx(self.graphs[i], node_attrs=['c','b','e'])

        #     data_matrix = th.cat((th.Tensor([self.properties[i]]*temp_graph.number_of_nodes()),
        #                                     temp_graph.in_degrees().view(-1, 1).float(), 
        #                                     temp_graph.ndata['c'].view(-1, 1).float(),
        #                                     temp_graph.ndata['b'].view(-1, 1).float(),
        #                                     temp_graph.ndata['e'].view(-1, 1).float()
        #                                     )
        #                                     , 1)
        #     if i == 0:
        #         print(data_matrix)
        #     temp_graph.ndata['s'] = data_matrix


            self.graphs[i] = temp_graph
            # # add self edges
            # nodes = self.graphs[i].nodes()
            # self.graphs[i].add_edges(nodes, nodes)

        # print(self.graphs[0].ndata['s'])

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
        # return self.graphs[idx], self.labels[idx], self.properties[idx]


class my_graph_Dataset_node_label(object):
    def __init__(self, data, labels):
        graphs, properties = map(list, zip(*data))
        self.graphs = graphs
        self.labels = labels
        # self.labels = th.Tensor(labels)

        # self.properties = np.asarray(properties)
        self.properties = np.asarray([list(p[:8])+list(p[12:14]) for p in properties])
        # self.properties = np.column_stack((self.properties[:,12], self.properties[:,13]))
        print("max diam",max(self.properties[:,5]*50), sep=': ')
        # only one of the following can be set to true
        use_only_id = True
        use_all_props = False

        # # preprocess
        for i in range(len(self.graphs)):
            temp_graph = dgl.DGLGraph()

            if use_only_id:
                temp_graph.from_networkx(self.graphs[i])
                node_ids = list(range(len(self.graphs[i].nodes)))
                np.random.shuffle(node_ids)

                # store random ID for each graph
                ndata = th.zeros((len(node_ids),1))
                for n_id, j in zip(node_ids, range(len(node_ids))):
                    ndata[j] = n_id
                # temp_graph.ndata['s'] = ndata

                nx.set_node_attributes(self.graphs[i], nx.betweenness_centrality(self.graphs[i]), 'b')
                nx.set_node_attributes(self.graphs[i], nx.closeness_centrality(self.graphs[i]), 'c')
                nx.set_node_attributes(self.graphs[i], nx.eigenvector_centrality_numpy(self.graphs[i]), 'e')

                temp_graph.from_networkx(self.graphs[i], node_attrs=['c','b','e'])

                data_matrix = th.cat((th.Tensor([self.properties[i]]*temp_graph.number_of_nodes()),
                                                temp_graph.in_degrees().view(-1, 1).float(), 
                                                temp_graph.ndata['c'].view(-1, 1).float(),
                                                temp_graph.ndata['b'].view(-1, 1).float(),
                                                temp_graph.ndata['e'].view(-1, 1).float(),
                                                ndata
                                                )
                                                , 1)
                # data_matrix = th.cat((temp_graph.in_degrees().view(-1, 1).float(), 
                #                                 ndata
                #                                 )
                #                                 , 1)
                temp_graph.ndata['s'] = data_matrix

                temp_label = self.labels[i]
                min_id =[len(temp_label)]* len(list(np.unique(temp_label)))
                for iter_node_index in range(len(temp_label)):
                    cluster_id = int(temp_label[iter_node_index])
                    random_id = int(node_ids[iter_node_index])
                    if min_id[cluster_id] > random_id:
                        min_id[cluster_id] = int(random_id)
                self.labels[i] = [min_id[int(cluster_id)] for cluster_id in temp_label]

            if use_all_props:
                nx.set_node_attributes(self.graphs[i], nx.betweenness_centrality(self.graphs[i]), 'b')
                nx.set_node_attributes(self.graphs[i], nx.closeness_centrality(self.graphs[i]), 'c')
                nx.set_node_attributes(self.graphs[i], nx.eigenvector_centrality_numpy(self.graphs[i]), 'e')
                temp_graph.from_networkx(self.graphs[i], node_attrs=['c','b','e'])

                data_matrix = th.cat((th.Tensor([self.properties[i]]*temp_graph.number_of_nodes()),
                                                temp_graph.in_degrees().view(-1, 1).float(), 
                                                temp_graph.ndata['c'].view(-1, 1).float(),
                                                temp_graph.ndata['b'].view(-1, 1).float(),
                                                temp_graph.ndata['e'].view(-1, 1).float()
                                                )
                                                , 1)

                temp_graph.ndata['s'] = data_matrix


            self.graphs[i] = temp_graph
            # # add self edges
            # nodes = self.graphs[i].nodes()
            # self.graphs[i].add_edges(nodes, nodes)

        print(self.graphs[0].ndata['s'])

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
        # return self.graphs[idx], self.labels[idx], self.properties[idx]