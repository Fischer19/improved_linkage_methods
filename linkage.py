import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def average_distance(set1, set2):
    dist = 0
    for i in set1:
        for j in set2:
            dist += np.linalg.norm(i - j)
    return dist / (len(set1) + len(set2))

class Linkage():
    def __init__(self, data, method = "Average"):
        self.data = data
        self.clusters = []
        self.c_history = []
        self.method = method
        self.index = data.shape[0]
        self.linkage_matrix = []

        for i, item in enumerate(data):
            self.clusters.append(([item], i))
            self.c_history.append([item])

    def run_linkage(self):
        flag = True
        while flag:
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
                    #print(c1, c2)
                    dist = average_distance(c1, c2)
                    if dist < min_dist:
                        min_dist = dist
                        c1_index = self.clusters[i][1]
                        c2_index = self.clusters[j][1]
                        pop_index1 = i
                        pop_index2 = j
            # create new cluster
            new_cluster = self.clusters[pop_index1][0] + self.clusters[pop_index2][0]
            c1 = self.clusters.pop(pop_index1)[0]
            c2 = self.clusters.pop(pop_index2 - 1)[0]
            self.clusters.append((new_cluster, len(self.c_history)))
            self.c_history.append(new_cluster)
            # log the merge information into linkage matrix
            self.linkage_matrix.append([c1_index, c2_index, min_dist, len(c1) + len(c2)])
            if len(self.clusters) == 1:
                flag = False
        return np.array(self.linkage_matrix)



test_data1 = np.random.normal(-1,1,100)
test_data2 = np.random.normal(1,1,100)
test_data = np.concatenate([test_data1, test_data2], axis = 0)
print(test_data.shape)

avl = Linkage(test_data)
Z = avl.run_linkage()
plt.figure(figsize=(25,10))
d = dendrogram(Z, leaf_font_size=10., leaf_rotation=0., get_leaves=True)
plt.show()