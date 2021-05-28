import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
# here = os.path.dirname(__file__)


def string_input(num):
    prop_file = './same_stat_4_tuple.o123281'
    # print(prop_file)
    prop_same_stat_dict = {}
    with open(prop_file, 'r') as prop_file_reader:
        lines = list(prop_file_reader.readlines())
        random.shuffle(lines)
        lines = lines[:num]
        for index, line in enumerate(lines):
            stat_string = line.rstrip('\n')
            if index == num:
                break
            if stat_string in prop_same_stat_dict:
                rep_stat_list = prop_same_stat_dict[stat_string]
                rep_stat_list.append(str(index))
                prop_same_stat_dict[stat_string] = rep_stat_list
            else:
                prop_same_stat_dict[stat_string] = [str(index)]
    return prop_same_stat_dict

def csv_input(num):
    prop_file = './ER_v=100.csv'
    # print(prop_file)
    prop_same_stat_dict = {}
    with open(prop_file, 'r') as prop_file_reader:
        lines = list(prop_file_reader.readlines())
        random.shuffle(lines)
        lines = lines[:num]
        for index, line in enumerate(lines):
            stat_string = line.rstrip('\n').split(',')[:14]
            stat_string.pop(1)
            stat_string = "".join(["{0:.2f} ".format(float(prop)) for prop in stat_string])
            if index == num:
                break
            if stat_string in prop_same_stat_dict:
                rep_stat_list = prop_same_stat_dict[stat_string]
                rep_stat_list.append(str(index))
                prop_same_stat_dict[stat_string] = rep_stat_list
            else:
                prop_same_stat_dict[stat_string] = [str(index)]
    return prop_same_stat_dict

def check_repetition(num):

    # prop_same_stat_dict = string_input(num)
    prop_same_stat_dict = csv_input(num)
    # print(prop_same_stat_dict)

    graph_repeat_set = {}
    for stat_string, index_list in prop_same_stat_dict.items():
        if len(index_list) not in graph_repeat_set:
            graph_repeat_set[len(index_list)] = 1
            print(len(index_list))
            print(index_list)
            print(stat_string)
        else:
            graph_repeat_set[len(index_list)] += 1

    # print(graph_repeat_set)
    return graph_repeat_set


def main():
    n_time_dict = {}
    repeat_times = 1
    for i in range(repeat_times):
        rep_dict = check_repetition(1000)
        # rep_list = sorted(rep_list, key=lambda X: X[0])
        for len_tuple, times in rep_dict.items():
            n_time_dict[len_tuple] = rep_dict[len_tuple] + n_time_dict.get(len_tuple,0)
    for len_tuple, times in n_time_dict.items():
        print(str(times/repeat_times)+ ' : '+str(len_tuple)+"-tuple")
        # ' has '++' graphs in average inside 1000 graphs over 100 times'
    # print(n_time_dict)

def draw_graph_by_index(index):
    g_list = nx.read_graph6('./ER_v=100.g6')
    graph = g_list[index]
    nx.draw(graph,pos=nx.spring_layout(graph))
    plt.show()
    return graph


if __name__ == '__main__':
    main()
    draw_graph_by_index(12)
    draw_graph_by_index(391)
    draw_graph_by_index(786)
    draw_graph_by_index(888)
    