#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import os

import dgl
import networkx as nx
from dgl.data import DGLDataset
from dgl.data.citation_graph import (CiteseerGraphDataset, CoraGraphDataset,
                                     PubmedGraphDataset)
from dgl.data.gdelt import GDELTDataset
from dgl.data.gnn_benckmark import (AmazonCoBuyComputerDataset,
                                    CoauthorCSDataset)
from dgl.data.icews18 import ICEWS18Dataset
from dgl.data.knowledge_graph import FB15k237Dataset, FB15kDataset, WN18Dataset
from dgl.data.ppi import PPIDataset
from dgl.data.reddit import RedditDataset
from dgl.data.utils import load_graphs, save_graphs

from utils import dataset_utils


class RefactorCoauthorCSDataset(CoauthorCSDataset):
    """ 'Computer Science (CS)' part of the Coauthor dataset for node
    classification task.

    Copy of DGL CoauthorCSDataset class. Simply overwriting:
    - the process and load methods to create train, validation and test masks.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(CoauthorCSDataset, self).__init__(name='coauthor_cs', raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def process(self):
        npz_path = os.path.join(self.raw_path, self.name + '.npz')
        g = self._load_npz(npz_path)
        self._graph = g
        dataset_utils.create_train_val_test_split_mask(self._graph)
        self._data = [g]
        self._print_info()
        self.graph = self._graph
        
    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        dataset_utils.create_train_val_test_split_mask(self._graph)
        self._data = [graphs[0]]
        self._print_info()
        self.graph = self._graph 


class RefactorAmazonCoBuyComputerDataset(AmazonCoBuyComputerDataset):
    """ 'Computer' part of the AmazonCoBuy dataset for node classification
    task.

    Copy of DGL AmazonCoBuyComputerDataset class. Simply overwriting:.
    - the process and load methods to create train, validation and test masks.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(AmazonCoBuyComputerDataset, self).__init__(name="amazon_co_buy_computer", raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    # Overwrite num classes from library since do not correspond to labels
    @property
    def num_classes(self):
        return 10

    def process(self):
        npz_path = os.path.join(self.raw_path, self.name + '.npz')
        g = self._load_npz(npz_path)
        self._graph = g
        dataset_utils.create_train_val_test_split_mask(self._graph)
        self._data = [g]
        self._print_info()
        self.graph = self._graph

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self.graph = self._graph
        self._data = [graphs[0]]
        self._print_info()
        dataset_utils.create_train_val_test_split_mask(self._graph)


class RefactorCiteseerGraphDataset(CiteseerGraphDataset):
    """Citeseer citation network dataset.

    Copy of DGL CiteseerGraphDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorCiteseerGraphDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorCoraGraphDataset(CoraGraphDataset):
    """ Cora citation network dataset.

    Copy of DGL CoraGraphDataset class. 

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorCoraGraphDataset, self).__init__(raw_dir, force_reload, verbose)


class RefactorPubmedGraphDataset(PubmedGraphDataset):
    """ Pubmet citation network dataset.

    Copy of DGL PubmedGraphDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorPubmedGraphDataset, self).__init__(raw_dir, force_reload, verbose)


class RefactorGDELTDataset(GDELTDataset):
    """ GDELT dataset for event-based temporal graph.

    Copy of DGL GDELTDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorGDELTDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorFB15kDataset(FB15kDataset):
    """ FB15k link prediction dataset.

    Copy of DGL FB15kDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorFB15kDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorFB15k237Dataset(FB15k237Dataset):
    """ FB15k237 link prediction dataset.

    Copy of DGL FB15k237Dataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(FB15k237Dataset, self).__init__(name="FB15k-237", reverse=reverse, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorWN18Dataset(WN18Dataset):
    """ WN18 link prediction dataset.

    Copy of DGL WN18Dataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(WN18Dataset, self).__init__(name="wn18", reverse=True, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorICEWS18Dataset(ICEWS18Dataset):
    """ ICEWS18 dataset for temporal graph.

    Copy of DGL ICEWS18Dataset class. 

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorICEWS18Dataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)


class RefactorPPIDataset(PPIDataset):
    """ Protein-Protein Interaction dataset for inductive node classification.

    Copy of DGL PPIDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """
    def __init__(self, mode="train", raw_dir=None, force_reload=False, verbose=False):
        super(RefactorPPIDataset, self).__init__(mode, raw_dir, force_reload, verbose)


class RefactorRedditDataset(RedditDataset):
    """ Reddit dataset for community detection (node classification).

    Copy of DGL RedditDataset class.

    Args:
        raw_dir (str, optional): raw file directory that contains the input
            data directory. Defaults to None.
        force_reload (bool, optional): whether to reload the dataset. Defaults
            to False.
        verbose (bool, optional): whether to print out progress information.
            Defaults to True.
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, *args, **kwargs):
        super(RefactorRedditDataset, self).__init__(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
