#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import os
import torch as th
import numpy as np
from .graphs_utils import generate_mask_tensor


def create_train_val_test_split_mask(graph,
                                     train_size=0.6,
                                     val_size=0.2,
                                     test_size=0.2,
                                     respect_class_proportions=False,
                                     filter_on=None):
    """Creates train, val and test nodes masks from graph according to ratios.

    Note: Nodes mask are stored in graphndata:
    "train_mask", "val_mask", "test_mask"

    Args:
        graph (DGLGraph): a graph in DGL format.
        train_size (float: [0, 1], optional): portion of nodes to use for
            training. Defaults to 0.6.
        val_size (float: [0, 1], optional): portion of nodes to use for
            validation. Defaults to 0.2.
        test_size (float: [0, 1], optional): portion of nodes to use for
            testing. Defaults to 0.2.
        respect_class_proportions (bool, optional): whether to respect class
            proportion in sampling. Defaults to False.
        filter_on (str, optional): nodes attribute tp filter nodes before
            creating sets. Defaults to None.

    Raises:
        NotImplementedError: does not support True as respect_class_proportion
            for now.
    """
    num_samples = graph.number_of_nodes()
    if filter_on is None:
        available_indexes = list(range(num_samples))
    else:
        available_indexes = th.nonzero(
            graph.ndata[filter_on]).squeeze().tolist()
    total_available_indexes = len(available_indexes)
    if respect_class_proportions:
        raise NotImplementedError
    else:
        train_indexes = np.random.choice(available_indexes,
                                         int(total_available_indexes *
                                             train_size),
                                         replace=False)
        set_train_indexes = set(train_indexes)
        available_indexes = [
            el for el in available_indexes if el not in set_train_indexes
        ]
        val_indexes = np.random.choice(available_indexes,
                                       int(total_available_indexes * val_size),
                                       replace=False)
        set_val_indexes = set(val_indexes)
        available_indexes = [
            el for el in available_indexes if el not in set_val_indexes
        ]
        if train_size + val_size + test_size < 1:
            test_indexes = np.random.choice(available_indexes,
                                            int(total_available_indexes *
                                                test_size),
                                            replace=False)
        else:
            test_indexes = available_indexes

    graph.ndata["train_mask"] = generate_mask_tensor(train_indexes,
                                                     num_samples)
    graph.ndata["val_mask"] = generate_mask_tensor(val_indexes, num_samples)
    graph.ndata["test_mask"] = generate_mask_tensor(test_indexes, num_samples)
