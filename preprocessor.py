#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import dgl
from networkx.utils.decorators import not_implemented_for
import torch as th
import numpy as np
from utils import graphs_utils
import tqdm


class Preprocessor(object):
    """Base graph preprocessor class from which preprocessors are derived to
    transform litterature graphs with transactional graphs properties.

    Part of nodes will be considered as clients: everything is known, the other
    part is considered as non-clients: part of information (nodes features
    and/or edges are missing).

    Args:
            g (DGLGraph): a graph to transform.
            ratio (float): percentage of nodes to consider as clients.
    """

    def __init__(self):
        pass

    def __call__(self, g: dgl.DGLGraph, ratio: float, *args, **kwargs) -> dgl.DGLGraph:
        g = self._process(g, ratio, *args, **kwargs)
        return g

    @staticmethod
    def _sample_nodes(g, ratio):
        """Randomly samples a proportion of nodes according to ratio.

        Args:
            g (DGLGraph): a graph to sample nodes from.
            ratio (float): percentage of nodes to sample.

        Returns:
            (tuple):
                - (numpy array): indexes of sampled nodes.
                - (numpy array): corresponding nodes.
                - (numpy array): indexes of remaining nodes not sampled.
                Complementary of first output.
        """
        sample_size = max(1, int(ratio*g.num_nodes()))
        sampled_indexes = np.random.choice(
            list(range(g.num_nodes())), size=sample_size, replace=False)
        not_sampled_indexes = np.setdiff1d(
            np.array(range(g.num_nodes())), sampled_indexes)
        sampled_nodes = g.nodes()[sampled_indexes]
        return sampled_indexes, sampled_nodes, not_sampled_indexes

    @staticmethod
    def _sample_features(g, ratio_features):
        """Randomly samples a proportion of nodes features according to
        ratio_features.

        Args:
            g (DGLGraph): a graph to sample nodes features from.
            ratio_features (float): percentage of nodes to modify.

        Returns:
            (numpy array): indexes of sampled nodes features.
        """
        n_features = g.ndata["feat"].shape[1]
        sample_size = max(1, int(ratio_features*n_features))
        sampled_indexes = np.random.choice(
            list(range(n_features)), size=sample_size, replace=False)
        return sampled_indexes

    @staticmethod
    def _create_ndata_client_mask(g, sampled_indexes):
        """Creates mask from sampled_indexes and stores it in g.ndata.

        Note: Client node potentially isolated
        
        Args:
            g (DGLGraph): a graph where nodes are sampled.
            sampled_indexes (numpy array): indexes of sampled nodes.

        Returns:
            (DGLGraph): a graph with an additional feature "client" in g.ndata.
        """
        
        g.ndata["client"] = th.zeros(g.num_nodes(), dtype=th.int8)
        g.ndata["client"][sampled_indexes] = th.ones(
            len(sampled_indexes), dtype=th.int8)
        return g

    def _process(g, ratio, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _sample_features(g, sampled_indexes, not_sampled_indexes, sampled_features, method):
        """Nodes features modification.

        Args:
            g (DGLGraph): a graph to modify nodes features from.
            sampled_indexes (numpy array): indexes of sampled nodes (client nodes).
            not_sampled_indexes (numpy array): indexes of remaining nodes not sampled.
            sampled_features (numpy array): indexes of sampled nodes features.
            method (str, optional): type of modification to apply.
                Must be in ["zeros", "random", "drop", "imputation"].
                Defaults to "drop".

        Returns:
            (DGLGraph): preprocessed graph.
        """
        assert method in ["zeros", "random", "drop", "imputation"]
        if len(sampled_features) == g.ndata["feat"].shape[1]:
            g.ndata["feat"] = th.eye(g.num_nodes())

        else:
            line_mask = graphs_utils.generate_mask_tensor(
                sampled_features, g.ndata["feat"].shape[1])
            mask = th.zeros(g.ndata["feat"].shape, dtype=th.bool)
            mask[not_sampled_indexes] = line_mask

            if method == "zeros":
                blurred_feat = th.zeros(
                    len(not_sampled_indexes)*len(sampled_features))
                g.ndata["feat"][mask] = blurred_feat

            elif method == "random":
                blurred_feat = th.randn(
                    len(not_sampled_indexes)*len(sampled_features))
                g.ndata["feat"][mask] = blurred_feat

            elif method == "imputation":
                mean = g.ndata["feat"][sampled_indexes].mean(axis=0)
                g.ndata["feat"][mask] = mean[line_mask].repeat(
                    len(not_sampled_indexes))

            elif method == "drop":
                g.ndata["feat"] = g.ndata["feat"][:, line_mask == 0]

        return g

class SamplerPreprocessor(Preprocessor):
    """Sampler preprocessor derived from Preprocessor.

    Samples a proportion of nodes and removes edges not connected to the
    sampled nodes.
    It removes edges between non-client nodes.

    Usage:
    processor = SamplerPreprocessor()
    processed_graph = processor(g, ratio=.1)
    """
    def __init__(self):
        super(SamplerPreprocessor, self).__init__()

    def _process(self, g, ratio, remove_isolated_nodes=False, *args, **kwargs):
        """Process g to return a graph only with edges connected to a set of
        nodes sampled according to ratio.

        Args:
            g (DGLGraph): a graph to preprocess.
            ratio (float): percentage of nodes to consider as clients.
            remove_isolated_nodes (bool, optional): whether to remove isolated
                nodes from g. Defaults to False.

        Returns:
            (tuple):
                - (DGLGraph): preprocessed graph.
                - (tuple of tensors): removed edges.
        """
        g, sampled_indexes, sampled_nodes, _, removed_edges_src, removed_edges_dst = self._remove_non_client_edges(
            g, ratio)
        isolated_indexes = graphs_utils.get_isolated_nodes(g)
        if remove_isolated_nodes:
            g = self._remove_nodes_and_upt_masks(
                g, isolated_indexes, sampled_indexes)
        else:
            g = self._update_training_mask(
                g, isolated_indexes, sampled_indexes)
        return g, removed_edges_src, removed_edges_dst

    def _remove_non_client_edges(self, g, ratio):
        """
        Removes all edges between non-clients nodes.

        Args:
            g (DGLGraph): a graph to preprocess.
            ratio (float): percentage of nodes to consider as clients.

        Returns:
            (tuple):
                - (DGLGraph): preprocessed graph.
                - (numpy array): indexes of sampled nodes (client nodes).
                - (numpy array): corresponding nodes.
                - (numpy array): indexes of remaining nodes not sampled
                (non-client nodes). Complementary of first output.
                - (tuple of tensors): removed edges.
        """
        sampled_indexes, sampled_nodes, not_sampled_indexes = self._sample_nodes(g, ratio)
        edges = g.edges()

        src, dst = self._edge_filtering(edges, sampled_indexes, sampled_nodes)

        src, dst = th.tensor(src), th.tensor(dst)

        if len(src) > 0:
            eids = g.edge_ids(src, dst)
            g.remove_edges(eids)

        return g, sampled_indexes, sampled_nodes, not_sampled_indexes, src, dst

    @staticmethod
    def _edge_filtering(edges, sampled_iterr, sampled_nodes):
        if all(sampled_nodes.numpy() == sampled_iterr):
            sampled_iterr = set(sampled_iterr)
        else:
            sampled_iterr = sampled_nodes
        src = []
        dst = []
        for u, v in tqdm.tqdm(zip(*edges)):
            u, v = u.item(), v.item()
            if u not in sampled_iterr and v not in sampled_iterr:
                src.append(u)
                dst.append(v)
        return src, dst

    def _update_training_mask(self, g, isolated_indexes, sampled_indexes):
        """
        Removes isolated nodes from masks and creates a new mask
        corresponding to the sampled nodes to identify clients nodes.
        """
        if len(isolated_indexes) > 0 and "train_mask" in g.ndata.keys():
            g.ndata["train_mask"][isolated_indexes] = th.tensor([False]*len(isolated_indexes))
            g.ndata["val_mask"][isolated_indexes] = th.tensor([False]*len(isolated_indexes))
            g.ndata["test_mask"][isolated_indexes] = th.tensor([False]*len(isolated_indexes))

        g = self._create_ndata_client_mask(g, sampled_indexes)
        return g

    def _remove_nodes_and_upt_masks(self, g, isolated_indexes, sampled_indexes):
        """
        Removes isolated nodes from graph and creates a new mask
        corresponding to the sampled nodes to identify clients nodes.
        """
        g = self._create_ndata_client_mask(g, sampled_indexes)
        if len(isolated_indexes) > 0:
            g.remove_nodes(isolated_indexes)
        return g


class FeaturesSamplerPreprocessor(SamplerPreprocessor):
    """
    Add features sampling to SamplerPreprocessor.
    """
    def __init__(self):
        super(FeaturesSamplerPreprocessor, self).__init__()

    def _process(self, g, ratio, ratio_features, method="drop", remove_isolated_nodes=False):
        """Process g to return a graph only with edges connected to a set of
        nodes sampled according to ratio and with nodes features modified for the
        not sampled nodes.

        Args:
            g (DGLGraph): a graph to preprocess.
            ratio (float): percentage of nodes to consider as clients.
            ratio_features (float): percentage of nodes features to modify.
            method (str, optional): type of modification to apply.
                Must be in ["zeros", "random", "drop", "imputation"].
                Defaults to "drop".
            remove_isolated_nodes (bool, optional): whether to remove isolated
                nodes from g. Defaults to False.

        Returns:
        (tuple):
            - (DGLGraph): preprocessed graph.
            - (tuple of tensors): removed edges
        """
        g, sampled_indexes, sampled_nodes, not_sampled_indexes, removed_edges_src, removed_edges_dst = self._remove_non_client_edges(g, ratio)
        isolated_indexes = graphs_utils.get_isolated_nodes(g)
        if remove_isolated_nodes:
            g = self._remove_nodes_and_upt_masks(g, isolated_indexes, sampled_indexes)
        else:
            g = self._update_training_mask(g, isolated_indexes, sampled_indexes)
        if len(sampled_indexes) < g.num_nodes():
            sampled_features = self._sample_features(g, ratio_features)
            g = self._sample_features(g, sampled_indexes, not_sampled_indexes, sampled_features, method)
        return g, removed_edges_src, removed_edges_dst
