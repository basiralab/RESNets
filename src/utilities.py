import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

def show_mtrx(m, title = None):
    fig, ax = plt.subplots(figsize = (10, 5))

    min_val = int(m.min())
    max_val = int(m.max())

    cax = ax.matshow(m, cmap=plt.cm.seismic)
    fig.colorbar(cax, ticks=[min_val, int((min_val + max_val)/2), max_val])
    plt.title(title)
    plt.show()


def plot_results(MADs, MSEs):


    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    if len(MADs) == 3:
        mad = {"K": list(range(2, 2 + len(MADs[0]))), "xTx": MADs[0], "zTz": MADs[1], "rTr": MADs[2]}
    else:
        mad = {"K": list(range(2, 2 + len(MADs[0]))), "xTx": MADs[0], "zTz": MADs[1]}
    df_mad = pd.DataFrame(mad)
    melted = df_mad.melt(id_vars="K", var_name="technique")
    sns.barplot(x="K", y="value", hue="technique", data=melted, ax=axes[0])

    if len(MADs) == 3:
        mse = {"K": list(range(2, 2 + len(MSEs[0]))), "xTx": MSEs[0], "zTz": MSEs[1], "rTr": MSEs[2]}
    else:
        mse = {"K": list(range(2, 2 + len(MSEs[0]))), "xTx": MSEs[0], "zTz": MSEs[1]}

    df_mse = pd.DataFrame(mse)
    melted = df_mse.melt(id_vars="K", var_name="technique")
    sns.barplot(x="K", y="value", hue="technique", data=melted, ax=axes[1])
    plt.show()


def predict(S, G, k, subject = 67, length = 595):
    RESULT = np.zeros((subject, length))

    for sub in range(subject):
        pred = get_top_k(S[sub], k)

        result_vec = np.zeros((1, length))
        for index in pred:
            result_vec = np.add(result_vec, G[index])

        result_vec = result_vec/k
        RESULT[sub] = result_vec

    return RESULT


def to_2d(vector, n_r):
    size = n_r
    x = np.zeros((size, size))
    c = 0
    for i in range(0, size):
        for j in range(0, i):
            x[i, j] = vector[c]
            x[j, i] = vector[c]
            c = c + 1
    return x

#preprocess data
def preprocess(data):

    scaler = MinMaxScaler()
    temp = data.T
    temp = scaler.fit_transform(temp)

    return temp.T



#build embeddings
def build_embeddings(enc, data, n_subject = 67, n_reg = 35, isFlat = True):

    adjacency = np.zeros((n_subject, n_reg, n_reg))
    # transform the feature matrix to multiple adjacency matrices
    if isFlat:
        for i in range(0, n_subject,):
            adjacency[i] = to_2d(data[i], n_reg)
    else:
        adjacency = data

    features = np.identity(n_reg)
    result = np.zeros((n_subject, n_reg, 1))
    for i in range(0, n_subject):
        print(' ')
        print('Subject: ' + str(i + 1))
        print('---------------------')
        # learn the graph embedding for each subject
        encoded_view = enc.erun(csr_matrix(adjacency[i]), features)
        result[i] = encoded_view

    return np.reshape(result, (n_subject, n_reg))

#build similarity matrices
def build_similarity_matrix(data):
    temp = np.matmul(data, data.T) #square matrix
    return temp

def get_top_k(arr, k):
    idx = (-arr).argsort()[:k]
    return idx.tolist()
