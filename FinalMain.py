import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from scipy.stats import multivariate_normal
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x; pos[:, :, 1] = y
# rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
# plt.contourf(x, y, rv.pdf(pos))
# plt.show()

def main():
    #warmUp()
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
    print('Running lab...')
    # data = np.loadtxt('./bcdata.csv', delimiter=',')
    data = np.loadtxt('./data_banknote_authentication.csv', delimiter=',')
    X = data[:, :-1]

    '''
    PART A
    '''
    #Perform SVD and get first singular vector.
    U, s, V = np.linalg.svd(X)
    print('Singular vectors of A.')
    v1 = V[0, :]
    v1 = v1.reshape((v1.size, 1))
    pcaData = np.dot(v1.T, X.T).T


    '''
    PART B
    '''
    D1 = data[data[:, -1] == 0]
    D1 = D1[:, :-1]
    # Get data set for c2
    D2 = data[data[:, -1] == 1]
    D2 = D2[:, :-1]
    meanc1 = D1.mean(axis=0)
    meanc1 = meanc1.reshape((D1.shape[1], 1))
    meanc2 = D2.mean(axis=0)
    meanc2 = meanc2.reshape((D2.shape[1], 1))
    # Calculate B
    B = np.dot(meanc1 - meanc2, (meanc1 - meanc2).T)
    # Calculate centered class matricies.
    Z1 = D1 - np.dot(np.ones((D1.shape[0], 1)), meanc1.T)
    Z2 = D2 - np.dot(np.ones((D2.shape[0], 1)), meanc2.T)
    # Calculate scatter matricies Si
    S1 = np.dot(Z1.T, Z1)
    S2 = np.dot(Z2.T, Z2)
    # Calculate within class scatter matrix S.
    S = S1 + S2
    # Solve the reduced eigenvalue problem.
    eigVals, eigVects = sp_linalg.eig(B, S, left=True, right=False)
    # eigVals, eigVects = np.linalg.eig(np.dot(np.linalg.inv(S), B))
    w = eigVects[:, 0]
    w = w.reshape((w.size, 1))

    #Project onto the LDA space.
    ldaData = np.dot(w.T, X.T).T

    '''
    PART C
    '''
    #LS on PCA data.
    pcaX = np.hstack((pcaData, np.ones((pcaData.shape[0], 1))))
    pcaW = np.linalg.pinv(pcaX).dot(data[:, -1])
    pcaPredicted = []
    for row in pcaX:
        out = np.dot(pcaW, row)
        if out >= 0:
            pcaPredicted.append(1)
        else:
            pcaPredicted.append(0)
    pcaAccuracy = calculateAccuracy(data[:, -1], pcaPredicted)
    print('PCA LS classification accuracy: %.12f' % pcaAccuracy)
    #LS on LDA data.
    ldaX = np.hstack((ldaData, np.ones((ldaData.shape[0], 1))))
    ldaW = np.linalg.pinv(ldaX).dot(data[:, -1])
    ldaPredicted = []
    for row in ldaX:
        out = np.dot(ldaW, row)
        if out >= 0:
            ldaPredicted.append(1)
        else:
            ldaPredicted.append(0)
    ldaAccuracy = calculateAccuracy(data[:, -1], ldaPredicted)
    print('LDA LS classification accuracy: %.12f' % ldaAccuracy)

    '''
    PART D
    '''
    #PCA plot.
    X0 = pcaData[data[:, -1] == 0]
    X1 = pcaData[data[:, -1] == 1]
    plt.scatter(X0 , np.zeros(X0.size),color='r')
    plt.scatter(X1, np.ones(X1.size),color='b')
    plt.xlabel('1-D coordinate')
    plt.title('PCA projected data points.')
    plt.show()

    #LDA plot.
    X0 = ldaData[data[:, -1] == 0]
    X1 = ldaData[data[:, -1] == 1]
    plt.scatter(X0 , np.zeros(X0.size),color='r')
    plt.scatter(X1, np.ones(X1.size),color='b')
    plt.title('LDA projected data points.')
    plt.xlabel('1-D coordinate')
    plt.show()

    '''
    PART E
    '''
    c = 1.5
    thresholdPredicted = []
    for projPoint in ldaData:
        if projPoint <= c:
            thresholdPredicted.append(1)
        else:
            thresholdPredicted.append(0)
    thresholdAccuracy = calculateAccuracy(data[:, -1], thresholdPredicted)
    print('LDA threshold classification accuracy: %.12f' % thresholdAccuracy)




'''
Calculate straight number of times that two class labels agreed.
'''
def calculateAccuracy(yactual, ypredicted):
	metrics = {}
	metrics["accuracy"] = 0

	for i in range(0, len(yactual)):
		if ypredicted[i] == yactual[i]:
			metrics["accuracy"] += 1

	metrics["accuracy"] = metrics["accuracy"] / float(len(yactual))
	return metrics["accuracy"]



if __name__ == '__main__':
    main()



















