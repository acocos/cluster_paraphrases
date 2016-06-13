## Implementation of Semantic Clustering of Pivot Paraphrases (Apidianaki et al. 2014)

## Adapted from Marianna Apidianaki's code

import numpy as np
import networkx as nx

def mean_threshold(sims):
    cpt = sum(sum((sims>0) * sims))
    sc_sum = sum(sum(sims>0))
    if cpt > 0:
        return sc_sum / cpt
    else:
        return 0

def dyn_threshold(sims):
    ''' Compute dynamically a similarity score threshold
    :param sims: similarity scores (V x V)
    :return: float
    '''
    T = mean_threshold(sims)
    if T > 0 and T < 1:
        new_T = aux_dyn_threshold(T, sims)
        next_T = aux_dyn_threshold(new_T, sims)
        while new_T != next_T:
            tmp_T = next_T
            next_T = aux_dyn_threshold(next_T, sims)
            new_T = tmp_T
    else:
        next_T = 0
    return next_T


def aux_dyn_threshold(T, msim):
    below_T = []
    above_T = []
    N = len(msim)
    for i in range(N-1):
        for j in range(i+1, N):
            score = msim[i][j]
            if score > 0:
                if score >= T:
                    above_T.append(score)
                else:
                    below_T.append(score)
    return (np.mean(below_T) + np.mean(above_T))/2


def toGraph(msim, labels, grname, w2p):
    ''' Create a graph for a paraphrase set (paraphrase = nodes) and link two
    nodes based upon the connections in <w2p>.
    :param msim: adjacency matrix
    :param labels: labels for rows/columns of adjacency matrix
    :param grname: str
    :param w2p: dict of {word: {word: score}}
    :return:
    '''
    threshold = dyn_threshold(msim)

    gr = nx.Graph(name=grname)
    for lab in labels:
        gr.add_node(lab)
    for i in labels:
        for j in labels:
            if j in w2p[i]:
                if w2p[i][j] >= threshold:
                    gr.add_edge(i,j)
    return gr


