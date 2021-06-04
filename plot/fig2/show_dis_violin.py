import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# gt = ['gt_v='+str(x)+'_prop_connected' for x in range(4,8)]
# generated_filename = [8,9] + list(range(10,85,5))
# generated_filename = ['graphs_ER_'+str(x) for x in generated_filename]
# gt.extend(generated_filename)

generated_filename =  list(range(5,101,5))
# generated_filename = range(4,8)
x_label = generated_filename
generated_filename = ['graphs_ER_'+str(x) for x in generated_filename]
task_name = 'ER'
# generated_filename = ['graphs_'+task_name+'_'+str(x)+'_p=0.5' for x in generated_filename]

# gt = ['gt_v='+str(x)+'_prop_connected' for x in range(4,8)]

stats_label = ["GCC", "SCC", "APL", "r", "diam", "den", "Ce",
			"Cd", "Cc", "Cb", "Cei", "R_G", "\u03C1 "]



def draw_violin(title, label, data):

	plt.violinplot(data, range(len(data)), points=400, 
		# widths=0.3,
			showmeans=True, showextrema=True, showmedians=False)
	data_mean = [np.mean(row) for row in data]
	expected_line = None
	# coeff = np.polyfit(label, data_mean, deg=10)
	# p = np.poly1d(coeff)

	# p = 1/2
	# # p = 'log'
	# if title == "GCC":
	# 	if p == 1/2:
	# 		expected_line = [ p for n in label]
	# 	else:
	# 		expected_line = [ math.log(n)/n for n in label]

		
	# # if title == "SCC":
	# # 	expected_line = [ 1/float(n/4) for n in label]	
	# if title == "APL":
	# 	if p == 1/2:
	# 		expected_line = [ (math.log(n)/(math.log(n)*1/2))/((n+1)/3) for n in label]
	# 	# else:
	# 	# 	expected_line = [ (math.log(n)/(math.log(n)*(math.log(n)/n)))/((n+1)/3) for n in label]
	# 	# print([(n+1)/3 for n in range(90,100)])
	# 	# print([nx.average_shortest_path_length(nx.path_graph(v_num)) for v_num in range(90,100)])
	# if title == "r":
	# 	expected_line = [ 0 for n in label]
	# if title == "diam":
	# 	if p == 1/2:
	# 		expected_line = [ 2/float((n-1)) for n in label]
	# 	else:
	# 		expected_line = [ (math.log(n)/(math.log(n)*math.log(n)/n))/float((n-1)) for n in label]

	# if title == "den":
	# 	if p == 1/2:
	# 		expected_line = [ 1/2 for n in label]
	# 	else:
	# 		expected_line = [math.log(n)/n for n in label]	
	# # if title == "Ce":
	# # 	expected_line = [ 0.3 for n in label]

	# # if title == "Cd":
	# # 	expected_line = [ 1/(4*math.log(n)) for n in label]
	# # if title == "Cd":
	# # 	expected_line = [1/n for n in label]
	# # if title == "Cc":
	# # 	expected_line = [ 1/(4*math.log(n)) for n in label]
	# # if title == "Cb":
	# # 	expected_line = [ 1/(2*n) for n in label]
	# # if title == "Cei":
	# # 	expected_line = [ 3/(n-1) for n in label]
	# if title == "R_G":
	# 	if p==1/2:
	# 		expected_line = [ (n-1)/(n-1+2/(n-2)) for n in label]
	# 		print()
	# 	else:
	# 		expected_line = [ (n-1)/(n/(math.log(n)/n)) for n in label]
	# if title == "\u03C1 ":
	# 	if p == 1/2:
	# 		expected_line = [ max(p,p) for n in label]
	# 	else:
	# 		expected_line = [math.log(n)/n for n in label]

	# if expected_line is not None:
	# 	plt.plot(range(len(label)), expected_line ,color='red')
	# # else:
	# # 	plt.plot(range(len(label)), p(label) ,color='red')
	# 	print(coeff)	
	plt.xticks(range(len(label)), [str(x) for x in label], color='black')
	if title == "r":
		plt.ylim(-1.05,1.05)
	else:
		plt.ylim(-0.05,1.05)


def get_data(pos, files_list=generated_filename):
	data = []
	for filename in files_list:
		filesdata = np.loadtxt(filename+".csv", delimiter=",")
		# print(len(filesdata))
		# print(len(filesdata[0]))
		filesdata = np.transpose(filesdata)
		data.append(filesdata[pos])
	return data

def main():
	
	for i in range(len(stats_label)):
		# draw_violin(stats_label[i], x_label, get_data(i,gt))
		draw_violin(stats_label[i], x_label, get_data(i))
		plt.title(stats_label[i], fontsize=20)
		# plt.show()
		plt.savefig("./image/"+task_name+"_"+stats_label[i]+"_new.png")
		plt.gcf().clear()

main()