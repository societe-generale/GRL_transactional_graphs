#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import networkx as nx
import numpy as np
import dgl
import torch as th
from tqdm import tqdm


def remove_edges_connected_to_node(graph, node, clients):
    """Processes a graph to remove part of edges connected to a given node.

    It removes edges connected to the node defined by its ID,
    but keeping edges connected to subgraph defined by nodes in clients.

    Args:
        graph (NetworkX Graph): a graph in NetworkX format.
        node (int): ID of corresponding node.
        clients (list of int): list of nodes IDS.

    Returns:
        NetworkX Graph: processed graph.
    """
    edge_list = graph.edges(node)
    filtered_edge_list = list(filter(lambda t: t[1] not in clients, edge_list))
    graph.remove_edges_from(filtered_edge_list)
    return graph


def sample_client_nodes(graph, ratio):
    """Samples nodes from a graph according to ratio.

    Args:
        graph (NetworkX Graph): a graph in NetworkX format.
        ratio (float: [0, 1[]): portion of nodes for which the edges will be
            kept. Defaults to 0.01.

    Returns:
        np.array: num_nodes*ratio sampled nodes.
    """
    sample_size = max(1, int(ratio * graph.number_of_nodes()))
    return np.random.choice(graph.nodes(), size=sample_size, replace=False)


def remove_isolated_nodes(graph):
    """Processes a graph to remove isolated nodes.

    Args:
        graph (NetworkX Graph): a graph in NetworkX format.

    Returns:
        NetworkX Graph: processed graph
    """
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    return graph


def remove_non_client_edges(graph, ratio=0.01):
    """Processes a graph to remove all edges between a sampled portion of the
    graph defined by ratio.

    It first samples ratio*graph.number_of_nodes, take the remaining nodes
    and remove all edges within this community.
    Edges from this community to the sampled nodes are kept.

    Args:
        graph (NetworkX Graph): a graph in NetworkX format.
        ratio (float: [0, 1[): portion of nodes for which the edges will be
            kept. Defaults to 0.01.

    Returns:
        NetworkX Graph: processed graph.
    """
    clients = sample_client_nodes(graph, ratio)
    non_clients = [n for n in graph.nodes() if n not in clients]
    for c in non_clients:
        graph = remove_edges_connected_to_node(graph, c, clients)

    graph = remove_isolated_nodes(graph)
    return graph


def get_isolated_nodes(g):
    """Looks for the list of indexes of isolated nodes in a graph.

    Note: works with directed graphs.

    Args:
        g (DGLGraph): a graph in DGL format.

    Returns:
        list of int: list of indexes of isolated nodes.
    """
    _out = g.out_degrees()
    _in = g.in_degrees()
    indexes, *_ = th.where(th.logical_and(_out == 0, _in == 0))
    return indexes.tolist()


def generate_mask_tensor(indexes, num_indexes):
    """Generates mask of the size of num_indexes with True at positions
    given by indexes.

    Args:
        indexes (list or numpy array): indexes where True are needed.
        num_indexes (int): size of the output mask.

    Returns:
        torch.tensor: mask
    """
    mask = th.tensor([False] * num_indexes)
    mask[indexes] = True
    return mask


def create_train_val_test_split_mask_edges(graph,
                                           train_size=0.85,
                                           val_size=0.05,
                                           test_size=0.10,
                                           respect_class_proportion=False,
                                           verbose=1):
    """Creates train, val and test nodes masks from graph according to ratios.

    Note: Edges mask are stored in graph.edata:
    "train_mask", "val_mask", "test_mask"

    Args:
        edges (torch tensor): tensor containing src and dst nodes defining
            edges.
        train_size (float: [0, 1], optional): portion of edges to use for
            training. Defaults to 0.85.
        val_size (float: [0, 1], optional): portion of edges to use for
            validation. Defaults to 0.05.
        test_size (float: [0,p 1], optional): portion of edges to use for
            testing. Defaults to 0.1.
        respect_class_proportion (bool, optional): whether to respect class
            proportion in sampling. Defaults to False.
        verbose (int, optional): verbose to print progess. Defaults to 1.

    Raises:
        NotImplementedError: does not support True as respect_class_proportion
            for now.
    Returns:
        tuple: (torch tensor, torch tensor, torch tensor):
            indexes of edges for the different splits.
    """
    available_indexes = get_avail_indexes_w_networkx(graph, verbose=verbose)
    num_samples = len(available_indexes)
    if respect_class_proportion:
        raise NotImplementedError
    else:
        train_indexes = np.random.choice(available_indexes,
                                         int(num_samples * train_size),
                                         replace=False)
        set_train_indexes = set(train_indexes)
        available_indexes = [
            el for el in available_indexes if el not in set_train_indexes
        ]
        val_indexes = np.random.choice(available_indexes,
                                       int(num_samples * val_size),
                                       replace=False)
        set_val_indexes = set(val_indexes)
        available_indexes = [
            el for el in available_indexes if el not in set_val_indexes
        ]
        if train_size + val_size + test_size < 1:
            test_indexes = np.random.choice(available_indexes,
                                            int(num_samples * val_size),
                                            replace=False)
        else:
            test_indexes = np.array(available_indexes)

    # Update masks
    n_edges = graph.number_of_edges()
    graph.edata["train_mask"] = generate_mask_tensor(train_indexes, n_edges)
    graph.edata["val_mask"] = generate_mask_tensor(val_indexes, n_edges)
    graph.edata["test_mask"] = generate_mask_tensor(test_indexes, n_edges)

    return (th.from_numpy(train_indexes).long(),
            th.from_numpy(val_indexes).long(),
            th.from_numpy(test_indexes).long())


def construct_negative_graph(g, k, device):
    """Expresses negative samples as a graph.

    Note: inspired by GDL guide:
    https://docs.dgl.ai/en/latest/guide/training-link.html#guide-training-link-prediction

    Args:
        g (DGLGraph): a graph in DGL format.
        k (int): number of negative edges for one positive edge.
        device (torch device): working device.

    Returns:
        DGLGraph: a negative graph in DGL format.
    """
    src, dst = g.edges()
    neg_src = src.repeat_interleave(k)
    neg_src = neg_src.to(device) if "cuda" in device.type else neg_src
    neg_dst = th.randint(0, g.number_of_nodes(), (len(src) * k, ))
    neg_dst = neg_dst.to(device) if "cuda" in device.type else neg_dst
    return dgl.graph((neg_src, neg_dst), num_nodes=g.number_of_nodes())


def unsup_graph_preprocessor(graph,
                             train_size=0.85,
                             val_size=0.05,
                             test_size=0.10,
                             respect_class_proportion=False):
    """Preproccesing used to train unsupervised models.

    Samples edges according to ratios, removes val/test edges and creates
    val/test negative edges with similar proportion with positive edges
    in subsets.

    Args:
        graph (DGLGraph): a graph in DGL format.
        train_size (float: [0, 1], optional): portion of edges to use for
            training. Defaults to 0.85.
        val_size (float: [0, 1], optional): portion of edges to use for
            validation. Defaults to 0.05.
        test_size (float: [0,p 1], optional): portion of edges to use for
            testing. Defaults to 0.1.
        respect_class_proportion (bool, optional): whether to respect class
            proportion in sampling. Defaults to False.

    Returns:
        tuple: (DGLGraph, list[tuple], list[tuple], list[tuple], list[tuple]):
        graph with removed edges. (Note: used for clarity reason, but useless
        since operation performed inplace),
        validation edges: source and destination nodes indexes,
        testing edges: source and destination nodes indexes.
    """
    # Creation of edge masks
    _, val_indexes, test_indexes = create_train_val_test_split_mask_edges(
        graph, train_size, val_size, test_size, respect_class_proportion)

    # Remove validation and test edges and create lists for source/dest nodes
    # of val-test sets
    val_test_indexes = list(
        set(val_indexes.numpy()).union(set(test_indexes.numpy())))
    src_dst = np.array(list(zip(*graph.edges())))
    val_src_dst = src_dst[val_indexes]
    test_src_dst = src_dst[test_indexes]

    # Remove the opposite edges as DGL doubles edges
    opposite_indexes = get_opposite_index(src_dst, val_test_indexes)
    graph.remove_edges(val_test_indexes + opposite_indexes)

    neg_val_src_dst, neg_test_src_dst = sample_val_test_edges(
        graph.number_of_nodes(), src_dst, val_src_dst, test_src_dst)
    return graph, val_src_dst, test_src_dst, neg_val_src_dst, neg_test_src_dst


def get_avail_indexes_w_networkx(graph, verbose=1):
    """Transforms the graph into networkx and gets the list of edges indexes
    corresponding to unique edges (sets aside opposite edges as DGL doubles
    each edge). The use of networkx allows to have one edge over two.


    Args:
        graph (DGLGraph): a graph in DGL format.
        verbose (int, optional): verbose to print progess. Defaults to 1.

    Returns:
        list of int: list of edges indexes.
    """
    nx_edgeset = set(graph.to_networkx().to_undirected().edges())
    zip_edges = list(zip(*graph.edges()))
    zipped_edgelist = [(u.item(), v.item()) for u, v in zip_edges]

    output = []
    iter_edges = tqdm(
        enumerate(zipped_edgelist)) if verbose else enumerate(zipped_edgelist)
    for index, (src, dst) in iter_edges:
        if src == dst or (src, dst) in nx_edgeset:
            output.append(index)
        else:
            pass
    return output


def get_opposite_index(src_dst, edges_indexes):
    """Gets indexes of opposite edges from a list of edges indexes.

    Note: opposite as in DGL mindset opposite of a -> b is b -> a.

    Args:
        src_dst (numpy array): array of source/destination nodes
            representing edges.
        edges_indexes (list): list of egdes indexes from which the
            opposites are needed.

    Returns:
        list of int: list of indexes.
    """
    double_edges = [(elt[1], elt[0]) for elt in src_dst[edges_indexes]]
    src_dst_tuple = [(u.item(), v.item()) for u, v in src_dst]
    return [src_dst_tuple.index(elt) for elt in double_edges]


def sample_train_edges(graph, pos_train_src_dst, all_test_edges):
    """Samples negative edges for train set.

    Args:
        graph (DGLGraph): a graph in DGL format.
        pos_train_src_dst (numpy array): 2d array, list of positive train edges
            in format [src, dst].
        all_test_edges (numpy array): 2d array, list of positive and negative
            test edges in format [src, dst].

    Returns:
        negative_train_edges: 2D array, list of negative train edges
            in format [src, dst].
    """
    num_nodes = graph.number_of_nodes()
    num_edges = pos_train_src_dst.shape[0]

    pos_train_src_dst = set([(src, dst) for src, dst in pos_train_src_dst])
    all_test_edges = set([(src, dst) for src, dst in all_test_edges])

    neg_train_edges = set()
    while len(neg_train_edges) < num_edges:
        src, dst = np.random.randint(0, num_nodes), np.random.randint(
            0, num_nodes)
        if src == dst:
            continue
        if (src, dst) in pos_train_src_dst or (dst, src) in pos_train_src_dst:
            continue
        if (src, dst) in all_test_edges or (dst, src) in all_test_edges:
            continue
        neg_train_edges = neg_train_edges.union(set([(src, dst)]))
    return np.array([*neg_train_edges])


def sample_val_test_edges(num_nodes, all_edges, val_edges, test_edges):
    """Samples negative edges for validation and test sets.

    Args:
        num_nodes (int): number of nodes in graph.
        all_edges (numpy array): 2d array, list of all edges in graph
            in format [src, dst].
        val_edges (numpy array): 2d array, list of validation edges
            in format [src, dst].
        test_edges (numpy array): 2d array, list of test edges
            in format [src, dst].

    Returns:
        (numpy array), (numpy array): 2D arrays, list of negative val/test
            edges in format [src, dst].
    """
    all_edges_set = set([(src, dst) for src, dst in all_edges])

    neg_test_edges = set()

    len_test_edges = len(test_edges)
    while len(neg_test_edges) < len_test_edges:
        src, dst = np.random.randint(0, num_nodes), np.random.randint(
            0, num_nodes)
        if src == dst:
            continue
        if (src, dst) in all_edges_set or (dst, src) in all_edges_set:
            continue
        if (src, dst) in neg_test_edges or (dst, src) in neg_test_edges:
            continue
        neg_test_edges = neg_test_edges.union(set([(src, dst)]))

    neg_val_edges = set()
    while len(neg_val_edges) < len(val_edges):
        src, dst = np.random.randint(0, num_nodes), np.random.randint(
            0, num_nodes)
        if src == dst:
            continue
        if (src, dst) in all_edges_set or (dst, src) in all_edges_set:
            continue
        if (src, dst) in neg_test_edges or (dst, src) in neg_test_edges:
            continue
        if (src, dst) in neg_val_edges or (dst, src) in neg_val_edges:
            continue
        neg_val_edges = neg_val_edges.union(set([(src, dst)]))

    return np.array([*neg_val_edges]), np.array([*neg_test_edges])
