import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def average_distance(set1, set2):
    dist = 0
    for i in set1:
        for j in set2:
            dist += np.linalg.norm(i - j)
    return dist / (len(set1) * len(set2))

class Linkage():
    def __init__(self, data, k_init = None, threshold = 1, method = "Average"):
        self.data = data
        self.clusters = []
        self.c_history = []
        self.method = method
        self.index = data.shape[0]
        self.linkage_matrix = []
        self.distance = {}
        self.threshold = threshold
        self.k_init = k_init


        for i, item in enumerate(data):
            self.clusters.append(([item], i))
            self.c_history.append([item])
            self.distance[i] = {}
            
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters)):
                c1 = self.clusters[i][0]
                c2 = self.clusters[j][0]
                dist = average_distance(c1, c2)
                self.distance[i][j] = dist

    def run_linkage(self):
        # k-mean initiation
        pop_list = list(self.clusters[i] for i in self.k_init)
        for k_c in sorted(self.k_init, reverse=True):
            del self.clusters[k_c]
        for i, k_c in enumerate(self.k_init):
            global_index1 = k_c
            c1 = pop_list[i][0]
            min_dist = 1e10
            for j, d in enumerate(self.clusters):
                dist = average_distance(d[0], c1)
                if dist < min_dist:
                    pop_index2 = j
                    global_index2 = d[1]
                    min_dist = dist
            c2 = self.clusters.pop(pop_index2)[0]        
            new_cluster = c1 + c2
            self.clusters.append((new_cluster, len(self.c_history)))
            self.c_history.append(new_cluster)
            self.distance[len(self.c_history) - 1] = {}
            for i in range(len(self.clusters)):
                global_index = self.clusters[i][1]
                dist = average_distance(new_cluster, self.clusters[i][0])
                self.distance[len(self.c_history) - 1][global_index] = dist
                self.distance[global_index][len(self.c_history) - 1] = dist
            # log the merge information into linkage matrix
            #print(global_index1, global_index2)
            self.linkage_matrix.append([global_index1, global_index2, min_dist, len(c1) + len(c2)])
            
        flag = True
        while flag:
            #print(len(self.clusters))
            min_dist = 1e10
            c1_index = 0
            c2_index = 0
            pop_index1 = 0
            pop_index2 = 0
            # determine what to merge next 
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    c1 = self.clusters[i][0]
                    c2 = self.clusters[j][0]
                    if len(c1) < self.threshold and len(c2) < self.threshold:
                        continue
                    c1_global_index = self.clusters[i][1]
                    c2_global_index = self.clusters[j][1]
                    #print(c1, c2)
                    dist = self.distance[c1_global_index][c2_global_index]
                    if dist < min_dist:
                        min_dist = dist
                        c1_index = c1_global_index
                        c2_index = c2_global_index
                        pop_index1 = i
                        pop_index2 = j
            # create new cluster
            new_cluster = self.clusters[pop_index1][0] + self.clusters[pop_index2][0]
            c1 = self.clusters.pop(pop_index1)[0]
            c2 = self.clusters.pop(pop_index2 - 1)[0]
            self.clusters.append((new_cluster, len(self.c_history)))
            self.c_history.append(new_cluster)
            self.distance[len(self.c_history) - 1] = {}
            for i in range(len(self.clusters)):
                global_index = self.clusters[i][1]
                dist = average_distance(new_cluster, self.clusters[i][0])
                self.distance[len(self.c_history) - 1][global_index] = dist
                self.distance[global_index][len(self.c_history) - 1] = dist
            # log the merge information into linkage matrix
            self.linkage_matrix.append([c1_index, c2_index, min_dist, len(c1) + len(c2)])
            if len(self.clusters) == 1:
                flag = False
        return np.array(self.linkage_matrix)



#test_data1 = np.random.normal(-1,1,100)
#test_data2 = np.random.normal(1,1,100)
#test_data = np.concatenate([test_data1, test_data2], axis = 0)
#print(test_data.shape)

test_data = np.array([1,2,3,4,5,6,11,12,13,14,15,16])
avl = Linkage(test_data, [[2,3],[5,6], [7,8]], 2)
Z = avl.run_linkage()
plt.figure(figsize=(25,10))
d = dendrogram(Z, leaf_font_size=10., leaf_rotation=0., get_leaves=True)
plt.show()