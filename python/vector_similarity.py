import pandas as pd
import os
os.chdir("../Dataset/Second dataset")
import math
import matplotlib.pyplot as plt
from collections import Counter
class Well:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

def vector_len(v):
    return math.sqrt(sum([x*x for x in v]))

def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum([x*y for (x,y) in zip(v1, v2)])

def euclidean_distance_similarity(v1, v2):
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(v1, v2)))

def manhattan_distance(v1,v2):
    return sum(abs(a-b) for a,b in zip(v1,v2))

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (vector_len(v1) * vector_len(v2))

def plot_k_most_similar_to_well(dataSet, columns, well_name, k):
    plt.figure(figsize=(15,7))
    v = [ well.vector for well in dataSet if well.name == well_name][0]
    
    plt.subplot(221)
    plt.title('Base Well Features')
    for i in range(21):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5)
        
    plt.scatter(range(len(v)), v, label = str(well_name))
    for i, txt in enumerate(columns):
            plt.annotate(txt, (i, 0), rotation = -90, fontsize = 'x-small')

    plt.legend()
    dict ={}
    for i in dataSet:
        dict[i.name] = cosine_similarity(i.vector, v)
    d = Counter(dict)
    plt.subplot(222)
    plt.title('Cosine similarity')
    for i in range(21):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5) 
    for name, vector in d.most_common(k):
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.scatter(range(len(vector[0:])), vector[0:], label = str(name))
        
        plt.legend()


    dict ={}
    for i in dataSet:
        dict[i.name] = euclidean_distance_similarity(i.vector, v)
    d = Counter(dict)
    plt.subplot(223)
    plt.title('Euclidian similarity')
    for i in range(21):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5)
    for name, vector in (d.most_common()[:-k-1:-1]):
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.scatter(range(len(vector[0:])), vector[0:], label = str(name))
        
        plt.legend()

    dict ={}
    for i in dataSet:
        dict[i.name] = manhattan_distance(i.vector, v)
    d = Counter(dict)
    plt.subplot(224)
    plt.title('Manhatan similarity')
    for i in range(21):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5)
 
    
    for name, vector in d.most_common()[:-k-1:-1]:
        vector = [ well.vector for well in dataSet if well.name == name][0]
        plt.scatter(range(len(vector[0:])), vector[0:], label = str(name))
        
        plt.legend()
    plt.show()



def plot_features(dataSet, columns):
    plt.figure(figsize=(15,7))
    plt.title('Well Features')
    for i in range(21):
        plt.axvline(x=0.5+i, color = 'black', linewidth = 0.5)
        
    for item in dataSet:
        plt.scatter(range(len(item.vector)), item.vector)
    for i, txt in enumerate(columns):
            plt.annotate(txt, (i, 0), rotation = -90, fontsize = 'small')
    plt.show()

def readDataSet():
    file_name = '022_preprocessing_after_normalizing_values.xlsx'
    df = pd.read_excel(file_name, index=False)
    dataSet = []
    for i in range(df.shape[0]):
        well = str('well_' + str(int(df.iloc[i]['Id'])) + '_'+ str(int(df.iloc[i]['Level'])))
        wellVector = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).iloc[i].values)
        dataSet.append(Well(well, wellVector))
    columns = list(df.drop(columns = ["Id", "Longitude", "Latitude", "Level"]).columns)
    return dataSet, columns

if __name__ == '__main__':

    dataSet, columns = readDataSet()
    plot_k_most_similar_to_well(dataSet, columns, 'well_1_1', 10)
    #plot_features(dataSet, columns)



