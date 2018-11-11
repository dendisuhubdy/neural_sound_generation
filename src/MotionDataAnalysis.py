from numpy import genfromtxt

from pca import run_pca, run_pca_np

def main():
    my_data = genfromtxt('./results/joint_angle_data.csv', delimiter=',')
    print("PCA input shape is: ")
    print(my_data.shape)
    pca = run_pca(my_data)
    print("PCA output shape is: ")
    print(pca.shape)


if __name__ == "__main__":
    main()
