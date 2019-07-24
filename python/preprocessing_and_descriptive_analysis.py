import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("../Dataset/Second dataset")
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

def normalizeValues(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x = df.values
    col = df.columns
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = col
    df.to_excel('preprocessing_after_normalizing_values.xlsx')
    return df


def plot_corr(df,size=10):
    corr = df.corr()

    pl, ax = plt.subplots(figsize=(20, 15))
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                     linewidths=.05, annot_kws={"size": 7})
    hm.set_xticklabels(corr.columns, rotation = 75, fontsize = 8)
    hm.set_yticklabels(corr.columns, rotation = 30, fontsize =11)
    pl.subplots_adjust(top=0.93, left=0.15, bottom = 0.2)
    t= pl.suptitle('Well Attributes Correlation Heatmap', fontsize=14)
    pl.savefig('Correlation_Plot.jpg')
    plt.show()
    return corr


def featureCorrelationRanking(df):
    c = df.corr()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")
    so = pd.DataFrame(so.dropna())
    so.to_excel("feture_correlation_ranking2.xlsx")
    return so

def PCA_(X, n):
    from sklearn.decomposition import PCA
    pca = PCA(n_component = n)
    X = pca.fit(X).transform(X)
    return  pd.DataFrame(X)



def draw_pair_wise_scatter_plots(df):
    sns.pairplot(df)
    plt.show()

def number_of_optimal_k_means_classes(df):
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    data = df
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.savefig('Number of optimal classes')
    plt.show()

#__________________________________________________________
def k_means(df):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.drop(['Id', 'Level'], axis=1)#'Column Volume (ft3)'

    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = df, columns = ['pc1', 'pc2'])
    kmeans = KMeans(n_clusters=4, max_iter=1000).fit(df)
    sns.scatterplot(principalDf['pc1'], principalDf['pc2'], hue = kmeans.labels_)
    plt.legend()
    plt.show()



def plotByLocation(df, label):
    """
    plt.scatter(df["Longitude"], df["Latitude"], c = df[label])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(str(label) + ' relation with location')
    plt.legend(('y0','y1'))
    plt.show()
    """    
    sns.scatterplot(df["Longitude"], df["Latitude"], hue = df[label])
    plt.title(str(label) + ' relation with location')
    plt.legend()
    plt.savefig('2D_'+str(label))
    plt.show()













