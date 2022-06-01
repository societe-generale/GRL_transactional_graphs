#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import numpy as np


def generate_deepwalk_rw(graph, node, len_walk):
    """Generates walk of specified lenght from a node.

    Args:
        graph (DGLGraph): a graph in DGL format.
        node (int): starting node ID.
        len_walk (int): lenght of the walks to generate.

    Returns:
        list: list of nodes ID in the generated walk.
    """
    walk = [node]

    while len(walk) < len_walk:
        node_neighbours = [n for n in graph[node]]
        weights = []
        norm = 0
        if not node_neighbours:
            return [node]*len_walk
        for n in node_neighbours:
            edge = graph[n][node]
            if 'weight' in edge:
                weights.append(edge['weight'])
                norm += edge['weight']
            else:
                weights.append(1)
                norm += 1

        weights = np.array(weights) / norm
        walk.append(np.random.choice(a=node_neighbours, p=weights))

    return walk
