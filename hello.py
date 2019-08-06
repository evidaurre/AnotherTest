#%load_ext autoreload
#%autoreload 2
#%matplotlib inline
#
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
import umap

df=pd.read_csv("sample.csv")

import kmapper as km
#from kmapper.plotlyviz import plotlyviz


mapper = km.KeplerMapper(verbose=0)
projected_data = mapper.fit_transform(data, projection=umap.UMAP(n_neighbors=8,
                                                                 min_dist=0.65,
                                                                 n_components=2,
                                                                 metric='euclidean',
                                                                 random_state=3571))

# Get the simplicial complex
#scomplex = mapper.map(projected_data,
#                      clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
#                      coverer=km.Cover(35, 0.9))

#mapper.visualize(scomplex,color_function=projected_data,path_html="26june2019.html",
#                      title="make_circles(n_samples=10, noise=0.03, factor=0.3)")

##projected_data=mapper.fit_transform(data,projection=[0,1])
##graph = mapper.map(projected_data,data,clusterer=sklearn.cluster.KMeans(2))
mapper.visualize(graph,color_function=projected_data,path_html="hello.html",
                    title="make_circles(n_samples=10, noise=0.03, factor=0.3)")
#
##lens = mapper.fit_transform(data, projection=[0])
##mapper.visualize(graph,color_function=projected_data,path_html="hello2.html",
##                    title="make_circles(n_samples=10, noise=0.03, factor=0.3)")
#simplicial_complex = mapper.map(lens, X=data,
#                                cover=km.Cover(n_cubes=20, perc_overlap=0.1))
