from numpy import genfromtxt

from pca import run_pca

def main():
    my_data = genfromtxt('./joint_angle_data.csv', delimiter=',')
    print(my_data.shape)
    pca = run_pca(my_data)
    print(pca)
    print(pca.shape)

if __name__ == "__main__":
    main()
