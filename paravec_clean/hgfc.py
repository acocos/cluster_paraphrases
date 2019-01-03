## Anne Cocos, 2015

# Implementation of Hierarchical Graph Factorization Clustering
## 1. Kai Yu, Shipeng Yu, and Volker Tresp. Soft clustering on graphs. NIPS 18:1553,2006.
## 2. Lin Sun and Anna Korhonen. Hierarchical Verb Clustering using Graph Factorization. EMNLP 2011:1023-1033.

import numpy as np
from sklearn.metrics import silhouette_score
import networkx as nx

def gfact(W, m, rand_seed=None):
    '''
    Factorize W to obtain bipartite graph K with adjacency matrix B
    :param W: similarity matrix (n x n)
    :param m: number of clusters
    :return: adjacency matrix B (n x m), vector L (length m)
    '''
    if rand_seed is not None:
        np.random.seed(rand_seed)
    
    n = W.shape[0]
    H = np.random.rand(n,m) + 1.
    # H = H / H.sum(axis=0)[np.newaxis,:]  # normalize cols = 1
    L = np.diag(H.sum(axis=0))

    for i in range(50):
        denom = H.dot(L).dot(H.T)
        W_ = W/denom
        W_[np.isinf(W_)] = 1.0  # 0/0=1
        W_[np.isnan(W_)] = 0.0

        H_ = H * W_.dot(H).dot(L)
        # H_ = H * (W_*(1-e)).dot(H).dot(L)
        H_ /= H_.sum(axis=0)[np.newaxis,:]  # normalize cols = 1
        L_ = L * H.T.dot(W_).dot(H)
        L_ = L_ / (sum(sum(L_))/sum(sum(W)))

        H = H_
        L = L_

        W_log = np.log10(W_)
        W_log[np.isinf(W_log)] = 0.0
        # diver = sum(sum(W * W_log - W + denom))  # TODO: Implement stack to track convergence
        # print diver

    B = H.dot(L)
    B.round(4)
    L.round(4)
    return B, L


def hgfc(W, m=None, thresh=0.01, rand_seed=None):
    '''
    Perform Hierarchical Graph Factorization Clustering
    If m is not given, will continue until the number of non-zero
    elements in L is 1
    :param W: adjacency matrix
    :param m: number of clusters to find in first round
    :return: Final round B, list of Bs for each level, and list of cluster numbers (Ms)
    '''
    n = W.shape[0]
    if m is None:
        Ms = [n-1]
    else:
        Ms = [m]

    Bs = []
    Ts = []
    As = []

    cluscnt = n  # number of clusters on last round
    # e = E
    while cluscnt > 1:

        B, L = gfact(W, Ms[-1], rand_seed=rand_seed)  # factorize G_{l-1} to obtain bitartite graph K with adjacency matrix B_l
        Bs.append(B)

        D = np.diag(np.sum(B,axis=1))

        Di = np.linalg.inv(D)

        if cluscnt == n:  # get cluster assignments (eq 5)
            t = Di.dot(B)
        else:
            t = Ts[-1].dot(Di).dot(B)
        Ts.append(t)
        tmax = np.max(t,axis=1)
        a = t-tmax[:,np.newaxis] >= - thresh*tmax[:,np.newaxis]

        As.append(a)

        W = B.T.dot(Di).dot(B)  # build a graph G_l with similarity matrix W_l = B_l.T * D_l^-1 * B_l

        if (cluscnt == n):
            B_ = B
        else:
            B_ = B_.dot(Di).dot(B)
        cluscnt = sum(a.sum(axis=0) > 0)  # number of clusters with non-empty assignments
        if cluscnt > 1:
            Ms.append(cluscnt-1)

    return B_, Bs, Ms, Ts, As


def labeldict(a, wordlist):
    return {i: [w for w,b in zip(wordlist, a[:,i]) if b] for i in range(a.shape[1])}

def labels(a):
    return np.array([np.argmax(row) for row in a])

def flatten(l):
    return [item for sublist in l for item in sublist]

def h_cluster(wordlist, sims, distmat, thresh=0.01, rand_seed=None):

    B_, Bs, Ms, Ts, As = hgfc(sims, thresh=thresh, rand_seed=rand_seed)

    sil_coefs = []
    for i,a in enumerate(As):
        l = labels(a)
        if len(set(l)) > 2 and len(set(l)) < len(wordlist)-1:
            sil_coefs.append(silhouette_score(distmat, labels(a), metric='precomputed'))
        else:
            sil_coefs.append(0.0)
    
    ld = [labeldict(a,wordlist) for a in As]
    ld_filt, sil_coefs_filt = zip(*[(lbldct, sc) for lbldct, sc in zip(ld, sil_coefs) 
                                    if len(set(flatten(lbldct.values())))>0])
    
    return ld_filt, sil_coefs_filt


def h_cluster_tree(wordlist, sims, thresh=0.01, rand_seed=None):
    B_, Bs, Ms, Ts, As = hgfc(sims, thresh=thresh, rand_seed=None)

    # Create tree
    T = nx.DiGraph()

    # Add leaf nodes
    for i, w in enumerate(wordlist):
        T.add_node('0.%d'%i, mem=[w])

    for l, a in enumerate(Bs):
        for j in range(a.shape[1]):
            newnode = '%d.%d'%(l+1,j)
            T.add_node(newnode)
            newmems = set([])
            for i in range(a.shape[0]):
                maxprob = np.max(a[i])
                if np.abs(a[i][j]-maxprob) <= 0.05*maxprob:
                    fromnode = '%d.%d'%(l,i)
                    T.add_edge(fromnode, newnode, mem=[])
                    newmems |= set(T.node[fromnode]['mem'])
            T.node[newnode]['mem'] = list(newmems)
