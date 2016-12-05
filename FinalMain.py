import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    warmUp()
    lab()


def warmUp():
    data = np.loadtxt('./fisher.csv', delimiter=',')

    #PART A WARM UP.
    #Find eigenvalues and eigen-vectors.
    A = np.diag((1, 2, 3))
    print(A)
    w, v = np.linalg.eig(A)
    print('Eigenvectors of A.')
    print(v)
    print('Eigenvalues of A.')
    print(w)

    # PART B WARM UP.
    # Find singular vectors of A.
    A = np.diag((1, 2, 3))
    print(A)
    U, s, V = np.linalg.svd(A)
    print('Singular vectors of A.')
    print(V)

    # PART C WARM UP.
    #Get data set for c1
    D1 = data[data[:, -1] == -1]
    D1 = D1[:, :-1]
    #Get data set for c2
    D2 = data[data[:, -1] == 1]
    D2 = D2[:, :-1]
    meanc1 = D1.mean(axis=0)
    meanc1 = meanc1.reshape((D1.shape[1], 1))
    meanc2 = D2.mean(axis=0)
    meanc2 = meanc2.reshape((D2.shape[1], 1))
    #Calculate B
    B = np.dot(meanc1 - meanc2, (meanc1 - meanc2).T)
    #Calculate centered class matricies.
    Z1 = D1 - np.dot(np.ones((D1.shape[0],1)), meanc1.T)
    Z2 = D2 - np.dot(np.ones((D2.shape[0],1)), meanc2.T)
    #Calculate scatter matricies Si
    S1 = np.dot(Z1.T, Z1)
    S2 = np.dot(Z2.T, Z2)
    #Calculate within class scatter matrix S.
    S = S1 + S2
    #Solve the reduced eigenvalue problem.
    eigVals, eigVects = sp_linalg.eig(B, S, left=True, right=False)
    #eigVals, eigVects = np.linalg.eig(np.dot(np.linalg.inv(S), B))
    w = eigVects[:, 0]
    w = w.reshape((w.size, 1))

    xList = []
    yList = []
    dataMean = data[:, :-1].mean(axis=0)
    for c in np.linspace(-4, 4, 80):
        xList.append(c * w[0] + dataMean[0])
        yList.append(c * w[1] + dataMean[1])

    #Create plots
    plt.scatter(data[:50, 0], data[:50, 1], color='r')
    plt.scatter(data[50:, 0], data[50:, 1], color='b')
    plt.plot(xList, yList, color='black')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Reduced Iris LDA plot.')

    plt.show()


def lab():
    pass


if __name__ == '__main__':
    main()



















