from numpy import genfromtxt

from pca import run_pca, run_pca_np

def main():
    my_data = genfromtxt('./joint_angle_data.csv', delimiter=',')
    print(my_data.shape)
    pca = run_pca(my_data)


if __name__ == "__main__":
    main()
