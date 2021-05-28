from scipy import stats
import numpy as np

dics = ['s_'+str(i) for i in range(1,11)]

num_prop = 13
correlation_list = [[0 for i in range(len(filenames)*len(dics))] for j in range(int(num_prop*(num_prop+1)/2))]
times = 0
for v in range(10,100,10):
	filenames = ['graphs_ER_'+str(v)+'_'+str(x*100) for x in [1,2,4,8,16]]
	for file_name in filenames:
		for dic in dics:
			matrix = np.loadtxt(dic+"/"+file_name+".csv", delimiter=",")
			matrix = np.transpose(matrix)
			index = 0
			for i in range(len(matrix)):
				list1 = matrix[i]
				for j in range(len(matrix)):
					if j < i:
						continue
					list2 = matrix[j]
					if np.ptp(list1) == 0 or np.ptp(list2) == 0:
						corr = np.nan
					else:
						corr, p = stats.pearsonr(list1, list2)

					correlation_list[index][times] = corr
					index += 1
			times += 1

np.savetxt("correlation_ER"+str(v)+".csv", correlation_list, delimiter=",")