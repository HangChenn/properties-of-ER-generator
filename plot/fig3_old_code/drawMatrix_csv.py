import matplotlib.pyplot as plt
from scipy import stats
from scipy import spatial
import numpy as np
import math
import sys

# you can change configuration, the code will load and draw the first path without extension(./data/graph9
# ./data/graph8) will draw 2 matrix based on graph9 and graph8 from data folder

class drawMatrix:
    def __init__(self, lists=None):
        self.graph10 = False
        self.volumetest = False
        self.truthVolume = None
        self.color_f_plot = ["b", "r", "g", "c", "m", "y", "k"]
        self.color_f_plot = ["r", "g", "c", "m", "y", "k"]
        
        #                   GT,         UN,         GE,         WS,     BA
        # self.color_f_plot = ["#404040", "#e41a1c", "#377eb8", "#984ea3", "#4daf4a" ]
        # self.color_f_plot = ["#404040", "#4daf4a" ]

        # self.stats_label = ["GCC", "ACC", "SCC", "APL", "r", "diam", "den", "Rt", "Cv",
        #                      "Ce", "E_G_resist", "s_rad"]
        self.stats_label = ["GCC", "SCC", "APL", "r", "diam", "den", "Ce",
			"Cd", "Cc", "Cb", "Cei", "e_g_resist", "s_rad"]
        self.my_drawing_list = []

        if lists is not None:
            self.my_drawing_list.append(lists)
        else:
            if len(sys.argv) >= 2:
                my_drawing_list = sys.argv[1:len(sys.argv)]
                self.draw_multi_graph(my_drawing_list)


        self.draw_multi_graph(self.my_drawing_list)


    def draw_multi_graph(self, my_drawing_list):
        yMin, xMin = [],[]
        yMax, xMax = [],[]
        correlation = open("correlation.txt", "a")

        #### check the boundary of data, this is meanly for generator ####
        for times in range(len(my_drawing_list)):
            if times == 0 and self.graph10:
                continue
            filename = my_drawing_list[times]
            Matrix = np.loadtxt(filename+".csv", delimiter=",")
            Matrix = np.transpose(Matrix)
            # Matrix = Matrix[0:10]
            # assort = Matrix[4]
            # for x in range(len(assort)):
            #     if np.isnan(assort[x]):
            #         assort[x] = 0
            # Matrix[4] = assort

            index = 0
            for i in range(len(Matrix)):
                list1 = Matrix[i]

                for j in range(len(Matrix)):
                    list2 = Matrix[j]

                    if j < i:
                        continue
                    init = 0
                    if self.graph10:
                        init = 1
                    if times == init:
                        yMin.append(min(list1))
                        xMin.append(min(list2))
                        yMax.append(max(list1))
                        xMax.append(max(list2))
                    else:
                        if min(list1) < yMin[index]:
                            yMin[index] = min(list1)
                        if min(list2) < xMin[index]:
                            xMin[index] = min(list2)
                        if max(list1) > yMax[index]:
                            yMax[index] = max(list1)
                        if max(list2) > xMax[index]:
                            xMax[index] = max(list2)

                    index = index + 1
        #### end check boundary ####
        for times in range(len(my_drawing_list)):
            filename = my_drawing_list[times]
            Matrix = np.loadtxt(filename+".csv", delimiter=",")
            Matrix = np.transpose(Matrix)
            # Matrix = Matrix[0:10]
            # assort = Matrix[4]
            # for x in range(len(assort)):
            #     if np.isnan(assort[x]):
            #         assort[x] = 0
            # Matrix[4] = assort
            # Convex Hull
            if self.volumetest and times == 0:
                npMatrix = np.asarray(Matrix)
                testone = npMatrix.transpose()
                hull = spatial.ConvexHull(testone)
                self.truthVolume = hull.volume
                print("Ground Truth",self.truthVolume)
            if self.volumetest and times > 0:
                npMatrix = np.asarray(Matrix)
                testone = npMatrix.transpose()
                sample = spatial.ConvexHull(testone)
                plt.text(-10 * xMax[index], yscale * 10 - times * yscale * 1.6,
                         "{0:.2f}".format(sample.volume / self.truthVolume), color=self.color_f_plot[times], fontsize=20)
            print("number of graph: " + str(len(Matrix[0])))
            print(filename)
            index = 0
            correlation.write("\n")
            for i in range(len(Matrix)):
                list1 = Matrix[i]
                correlation.write("\n")
                for j in range(len(Matrix)):
                    list2 = Matrix[j]

                    if j < i:
                        continue
                    plt.subplot(len(Matrix), len(Matrix), j+i*len(Matrix)+1)
                    if i == j:
                        plt.xlabel(self.stats_label[j],fontsize=20)
                        plt.ylabel(self.stats_label[i],fontsize=20)

                    corr, p = stats.pearsonr(list1, list2)
                    correlation.write("{0:.2f}".format(corr))
                    correlation.write(" ")
                    if np.isnan(corr):
                        print("p value for "+str(i) +" "+str(j)+": " +str(p))
                    # yscale = (yMax[index]-yMin[index])/2
                    # xscale = (xMin[index]-xMin[index])/2
                    xscale=yscale=1
                    # plt.text(1 + xscale*0.07, 1 - times * 0.5*yscale, "{0:.2f}".format(corr), color= self.color_f_plot[times],fontsize=12)
                    plt.text(xMax[index] + xscale*0.07, yMax[index] - times * 0.5*yscale, "{0:.2f}".format(corr), color= self.color_f_plot[times],fontsize=12)
                    if i == 3:
                        plt.xlim(-1,1)
                    else:
                        plt.xlim(0,1)
                    if j == 3:
                        plt.ylim(-1,1)
                    else:
                        plt.ylim(0,1)

                    
                    #   #######
                    if i == int((len(Matrix)-1)/2) and j == int((len(Matrix)-1)/2):
                        plt.text(-6, -5-times*yscale*1.6,filename,color= self.color_f_plot[times],fontsize=20)
                    if self.graph10:
                        if times != 0:
                            plt.plot(list2,list1,color=self.color_f_plot[times],marker="o", linestyle="")
                    else:
                        plt.plot(list2, list1, color=self.color_f_plot[times],marker="o", linestyle="")

                    # plt.plot(list2, list1, self.color_f_plot[times] + "o")
                    index = index+1
                    # plt.plot(list2,list1,color_f_plot[times])
            correlation.write("\n")

        plt.subplots_adjust(left=0.03, bottom=0.07, right=0.97, top=0.98,hspace=0.5,wspace=0.47)
        plt.show()
        correlation.close()


def data_to_log(list_to_modify):
    for ii in range(len(list_to_modify)):
        if list_to_modify[ii] == 0:
            continue
        list_to_modify[ii] = math.log(list_to_modify[ii])

    return list_to_modify


if __name__ == "__main__":
    graph = drawMatrix()
