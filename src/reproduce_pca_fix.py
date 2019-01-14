import numpy as np
import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sPCA
import sys
import os

len_paths = len(sys.argv) - 2
paths = []
print('total',len_paths,'files input')
for i in range(len_paths):
    paths.append(sys.argv[i + 1])
    print('path',i,'=',paths[i])
path_out = sys.argv[len_paths + 1]
print('path_out =',path_out)
if os.path.isfile(path_out) or os.path.isdir(path_out):
    print('Error: path_out already exists!')
    quit()

csvs = []
for i in range(len_paths):
    csvs.append(pd.read_csv(paths[i]))

def normalize(data):
    data = data.values
    data_max = np.max(data)
    data_min = np.min(data)
    data_range = data_max - data_min
    data_normalized = (data - data_min) / data_range
    return data_normalized

# def PCA(data):
#     mu = np.mean(data, axis=0)
#     data = data - mu
#     eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
#     print('eigenvectors', eigenvectors)
#     print('eigenvalues', eigenvalues)
#     if eigenvectors[0, 0] < 0.0 and eigenvectors[0, 1] < 0.0 and eigenvectors[0, 2] < 0.0:
#         eigenvectors[0, 0] = -eigenvectors[0, 0]
#         eigenvectors[0, 1] = -eigenvectors[0, 1]
#         eigenvectors[0, 2] = -eigenvectors[0, 2]
#     projected_data = np.dot(data, eigenvectors)
#     sigma = projected_data.std(axis=0).mean()
#     return mu, eigenvectors, eigenvalues, V, projected_data, sigma

def unit_vector(array):
    norm = np.linalg.norm(array)
    return array / norm

#new_features = []

# fig = plt.figure()
for disease in range(14):
    # ax = fig.add_subplot(3, 5, disease + 1)
    # ax = fig.add_subplot(3, 5, disease + 1, projection='3d')
    # plt.title('Disease ' + str(disease))
    # draw data points
    # nor_x = normalize(csv_1.iloc[:, disease + 1]).reshape(-1, 1)
    # nor_y = normalize(csv_2.iloc[:, disease + 1]).reshape(-1, 1)
    # nor_z = normalize(csv_3.iloc[:, disease + 1]).reshape(-1, 1)
    # nor_w = normalize(csv_4.iloc[:, disease + 1]).reshape(-1, 1)
    all_data = []
    for i in range(len_paths):
        all_data.append(normalize(csvs[i].iloc[:, disease + 1]). reshape(-1, 1))
    

    # calculate PCA
    pairs = np.hstack(all_data)
    # mu, eigenvectors, eigenvalues, V, projected_data, sigma = PCA(pairs)
    pca = sPCA(n_components=len_paths)
    pca.fit(pairs)
    # print(pca.components_)
    eigenvectors = pca.components_
    mu = pca.mean_


    # draw principal axis
    # pca_axis = []
    # pca_x = np.linspace(-1, 1, num=200)
    # for s in pca_x:
    #     axis_p = mu + s * eigenvectors[0]
    #     pca_axis.append(axis_p)
    # # pca_y = (eigenvectors[0, 1] / eigenvectors[0, 0] ) * pca_x.T
    # # plt.scatter(pca_x + mu[0], pca_y + mu[1], s=1)
    # pca_axis = np.array(pca_axis)
    # print('pca_axis.shape', pca_axis.shape)
    # ax.scatter(pca_axis[:, 0], pca_axis[:, 1], pca_axis[:, 2], s=2)

    # draw arrows of principle components
    # for axis in eigenvectors:
    #     start, end = mu - 5 * sigma * axis, mu + 5 * sigma * axis
    #     ax.annotate(
    #         '', xy=end, xycoords='data',
    #         xytext=start, textcoords='data',
    #         arrowprops=dict(facecolor='red', width=2.0, headwidth=0))
    
    # draw projection
    projected = np.dot(eigenvectors[np.newaxis, 0, :], pairs.T) # UX
    csvs[0].iloc[:, disease + 1] = projected[0]
    #print(projected)
    # reproduce = np.dot(eigenvectors[np.newaxis, 0, :].T, projected) #U^T (UX)
    # plt.scatter(reproduce[0], reproduce[1], s=1)
    #displacement = (mu[1] - (eigenvectors[0, 1] / eigenvectors[0, 0]) * mu[0]) * eigenvectors[0, 0] / np.sqrt(eigenvectors[0, 0]**2 + eigenvectors[0, 1]**2)
    #proj_unit = np.dot(unit_vector(eigenvectors[np.newaxis, 0, :].T), projected) + displacement * unit_vector(np.array([-eigenvectors[0, 1], eigenvectors[0, 0]]))[np.newaxis,:].T
    #plt.scatter(proj_unit[0], proj_unit[1], s=1)

    # ax.set_aspect('equal')
    # ax.set_xlim(-0.05, 1.1)
    # ax.set_ylim(-0.05, 1.1)
    # ax.set_zlim(-0.05, 1.1)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #plt.scatter([mu[0]], [mu[1]], s=5)

    # ax_xy = fig.add_subplot(2, 2, 2)
    # ax_xy.scatter(nor_x, nor_y, s=1)
    # ax_xy.scatter(pca_axis[:, 0], pca_axis[:, 1], s=2)
    # ax_xy.set_xlabel('x')
    # ax_xy.set_ylabel('y')
    
    # ax_xz = fig.add_subplot(2, 2, 3)
    # ax_xz.scatter(nor_x, nor_z, s=1)
    # ax_xz.scatter(pca_axis[:, 0], pca_axis[:, 2], s=2)
    # ax_xz.set_xlabel('x')
    # ax_xz.set_ylabel('z')

    # ax_yz = fig.add_subplot(2, 2, 4)
    # ax_yz.scatter(nor_y, nor_z, s=1)
    # ax_yz.scatter(pca_axis[:, 1], pca_axis[:, 2], s=2)
    # ax_yz.set_xlabel('y')
    # ax_yz.set_ylabel('z')
# plt.show()


csvs[0].to_csv(path_out, index=False)
# nor_x = normalize(csv_1.iloc[:, 1].values)
# nor_y = normalize(csv_2.iloc[:, 1].values)
# pairs = np.dstack((nor_x, nor_y))[0]
# print(pairs.shape)
# mu, egvec, egval, V, p, sigma = PCA(pairs)
# print(mu)
# print(egvec)
# print(egval)