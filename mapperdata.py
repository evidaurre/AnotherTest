import numpy as np
import pandas as pd
import sklearn

df=pd.read_csv("tda18may.csv")

#df.iloc[:,:].str.replace(',', '').astype(float)

def genderAssignment(x):

    if x == "M":
        return 0
    if x == "F":
        return 1
    else:
        return 0.5

df.Gender = df.Gender.map(genderAssignment)

data = df[["Amount", "Industry Code", "DMA Code", "Gender"]]

for i in range(len(data.columns)):
    x=data.columns[i]
    data[x]=data[x].astype(str)
    data[x] = data[x].str.replace(',','')
    data[x]=data[x].astype(float)

print(data)

print(type(data.Amount))

#data = data.values

#now try to mapper
import kmapper as km

mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1]) # X-Y axis

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, nr_cubes=10)

# Visualize it
mapper.visualize(graph, path_html="manyiudata_may18.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
