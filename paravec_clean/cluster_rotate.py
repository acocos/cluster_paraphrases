import numpy as np
from copy import deepcopy
import math

EPS=2.2204e-16
def local_scaling(distances):
    N = distances.shape[0]
    # K = min(int(round(np.sqrt(N))), 7)
    # K = 7
    K = int(round(np.sqrt(N)))

    sigmas = [0.5]*N  # default=0.5
    for i, row in enumerate(distances):
        gt0 = [v for v in sorted(row)[:K] if v < 1]
        if len(gt0) > 0:
            sigmas[i] = gt0[-1]
    return sigmas

def local_scaling_sims(sims):
    N = sims.shape[0]
    K = min(int(round(np.sqrt(N))), 7)

    sigmas = [0.5]*N  # default=0.5
    for i, row in enumerate(sims):
        gt0 = [v for v in sorted(row, reverse=True)[:K] if v > 0]
        if len(gt0) > 0:
            sigmas[i] = gt0[-1]
    return sigmas

def affinities(distances, sigmas):
    N = len(sigmas)
    A = np.array([[np.exp(-np.power(distances[i][j],2)/(sigmas[i]*sigmas[j]))
                   if i!=j else 0.0 for j in range(N)] for i in range(N)])
    return A

def affinities_sims(sims, sigmas):
    N = len(sigmas)
    A = np.array([[sims[i][j]/(sigmas[i]*sigmas[j])
                   if i!=j else 0.0 for j in range(N)] for i in range(N)])
    return A


def laplacian(A):
    dd = 1/(sum(A)+EPS)
    dd = np.sqrt(dd)
    DD = np.diag(dd)
    L = np.dot(DD,A).dot(DD)
    return L


def evecs(L,nEvecs):
    '''
    Calculate eigenvectors, eigenvalues of laplacian L
    '''
    u,s,v = np.linalg.svd(L)
    return s[:nEvecs], u[:,:nEvecs]

def cluster_rotate(A, group_num, verbose=False):
    '''
    Cluster by rotating eigenvectors to align with the canonical
    coordinate system
    :param A: Affinity matrix
    :param group_num: array of group numbers to test
    :return: (clusts, best_group_index, Quality, Vr)
    '''
    # zero diagonal of A if not already
    for i in range(len(A)):
        A[i][i] = 0.0
    L = laplacian(A)
    evals, V = evecs(L, max(group_num))
    try:
        group_num.remove(1)
    except ValueError:
        pass
    group_num.sort()

    Vcurr = V[:,:group_num[0]]

    clusts = []
    quality = []
    Vr = []
    for g, gn in enumerate(group_num):
        if g > 0:
            Vcurr = np.hstack((Vr[g-1],np.array([V[:,gn-1]]).transpose()))
            # Vcurr = np.vstack((Vr[g-1], V[:,gn])).transpose()  # only 2 col at a time
        c,q,v = evrot(Vcurr, debug=verbose)
        clusts.append(c)
        quality.append(q)
        Vr.append(v)
    maxQ = max(quality)
    i = [abs(maxQ-q) <= 0.001 for q in quality]
    best_group_index = min([idx for idx, b in enumerate(i) if b])
    return clusts, best_group_index, quality, Vr

def evrot(Vcurr, debug=False):
    ''' Compute the gradient of the eigenvectors alignment quality
    by gradient descent

    Input:
        V - eigenvectors, each column is a vector

    Output:
        clusts - Resulting cluster alignment
        Quality - final quality
        Vr - rotated eigenvectors

    :param Vcurr:
    :param debug:
    :return: (clusts, Quality, Vr)
    '''
    if debug:
        print 'Finding optimal rotation for V'

    # get number and length of eigenvectors dimensions
    X = deepcopy(Vcurr)
    ndata, dim = X.shape
    print 'Got %d vectors of length %d' % (dim, ndata)

    # get number of angles
    angle_num = (dim*(dim-1)/2)
    theta = np.zeros(angle_num)
    if debug:
        print 'Angle number is %d' % angle_num

    # build index mapping
    ik = [0] * angle_num
    jk = [0] * angle_num
    k = 0
    for i in range(dim-1):
        for j in range(i+1,dim):
            ik[k] = i
            jk[k] = j
            k += 1
    if debug:
        print 'Built index mapping for %d angles' % len(jk)
        print 'ik:', ik
        print 'jk:', jk

    # definitions
    max_iter = 200
    theta_new = deepcopy(theta)

    Q = evqual(X, dim, ndata)
    if debug:
        print 'Q = ', Q

    Q_old1 = Q
    Q_old2 = Q
    itr = 0
    alpha = 1.0

    while itr < max_iter:
        itr += 1
        for d in range(angle_num):
            dQ = evqualitygrad(X, theta, ik, jk, angle_num, d, dim, ndata)
            if debug:
                print 'gradient =', dQ
            theta_new[d] = theta[d] - alpha * dQ
            Xrot = rotate_givens(X, theta_new, ik, jk, angle_num, dim)
            Q_new = evqual(Xrot, dim, ndata)
            if Q_new > Q:
                theta[d] = theta_new[d]
                Q = Q_new
            else:
                theta_new[d] = theta[d]
        if iter > 2:
            if Q - Q_old2 < .001:
                break
        Q_old2 = Q_old1
        Q_old1 = Q

    print 'Done after %d iterations, Quality is %0.4f' % (itr, Q)

    Xrot = rotate_givens(X, theta_new, ik, jk, angle_num, dim)
    clusts = cluster_assign(Xrot, dim)

    # prepare output
    return clusts, Q, Xrot


def evqual(X, dim, ndata):
    ''' Evaluate alignment quality '''
    # take the square of all entries and find max of each row
    max_values = [max(r*r) for r in X]

    # compute cost
    J = sum([1.0/(max_values[i]+EPS) * sum(row * row) for i, row in enumerate(X)])
    J = 1.0 - (J/ndata - 1.0)/dim
    return J


def evqualitygrad(X, theta, ik, jk, angle_num, angle_index, dim, ndata):
    ''' Quality gradient '''

    # build V, U, A
    V = gradU(theta, angle_index, ik, jk, dim)
    U1 = build_Uab(theta, 0, angle_index-1, ik, jk, dim)
    U2 = build_Uab(theta, angle_index+1, angle_num-1, ik, jk, dim)
    A = buildA(X, U1, V, U2)

    # destroy no longer needed arrays
    del V
    del U1
    del U2

    # rotate vecs according to current angles
    Y = rotate_givens(X, theta, ik, jk, angle_num, dim)

    # find max magnitude of each row
    max_index = [np.argmax(np.abs(r)) for r in Y]
    max_values = [Y[i][m] for i,m in enumerate(max_index)]

    # compute gradient
    ind = 0
    dJ = 0
    for j in range(dim):
        for i in range(ndata):
            tmp1 = A[i][j] * Y[i][j] / np.power(max_values[i], 2)
            tmp2 = A[i][max_index[i]] * (Y[i][j]*Y[i][j]) / \
                   np.power(max_values[i], 3)
            dJ += tmp1 - tmp2
            # print i, j, dJ
    dJ = 2*dJ/ndata/dim
    return dJ


def rotate_givens(X, theta, ik, jk, angle_num, dim):
    G = build_Uab(theta, 0, angle_num-1, ik, jk, dim)
    return np.dot(X,G)

def cluster_assign(Xrot, dim):
    X = deepcopy(Xrot)
    # take square of all entries and find max of each row
    max_index = [np.argmax(r*r) for r in X]

    # prepare cluster assignments
    cluster_cell_array = [[e for e,m in enumerate(max_index) if m == i]
                          for i in range(dim)]

    return cluster_cell_array

def build_Uab(theta, a, b, ik, jk, dim):
    ''' Give rotation for angles a to b '''
    # print 'ik:', ik
    # print 'jk:', jk
    Uab = np.eye(dim)
    if b < a:
        return Uab
    for k in range(a,b+1):
        tt = theta[k]
        for i in range(dim):
            # u_ik = Uab[ik[i]][jk[i]] * math.cos(tt) - Uab[jk[i]][ik[i]] * math.sin(tt)
            # Uab[jk[i]][ik[i]] = Uab[ik[i]][jk[i]] * math.sin(tt) + Uab[jk[i]][ik[i]] * math.cos(tt)
            # Uab[ik[i]][jk[i]] = u_ik
            u_ik = Uab[i][ik[k]] * math.cos(tt) - Uab[i][jk[k]] * math.sin(tt)
            Uab[i][jk[k]] = Uab[i][ik[k]] * math.sin(tt) + Uab[i][jk[k]] * math.cos(tt)
            Uab[i][ik[k]] = u_ik
    return Uab


def gradU(theta, k, ik, jk, dim):
    ''' Gradient of a single Givens rotation '''
    V = np.zeros([dim,dim])

    V[ik[k]][ik[k]] = -math.sin(theta[k])
    V[ik[k]][jk[k]] = math.cos(theta[k])
    V[jk[k]][ik[k]] = -math.cos(theta[k])
    V[jk[k]][jk[k]] = -math.sin(theta[k])

    return V


def buildA(X, U1, Vk, U2):
    ''' X * U1 * Vk * U2 '''
    return np.dot(X, np.dot(U1, np.dot(Vk, U2)))
