from numpy import genfromtxt

from pca import run_pca, run_pca_np

def load_pca():
    my_data = genfromtxt('./results/joint_angle_data.csv', delimiter=',')
    # print("PCA input shape is: ")
    # print(my_data)
    # print(my_data.shape)
    # remember to do a transpose on the input
    pca_matrix = run_pca(my_data.T)
    return pca_matrix
    # print("PCA output shape is: ")
    # print(pca)
    # print(pca.shape)


if __name__ == "__main__":
    pca_matrix = load_pca()
