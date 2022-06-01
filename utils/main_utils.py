#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import os
import random
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from preprocessor import (FeaturesSamplerPreprocessor, SamplerPreprocessor)

from utils.graphs_utils import (create_train_val_test_split_mask_edges,
                                unsup_graph_preprocessor)
from utils.experiments_utils import load_config, save_json, read_json
from copy import deepcopy
import uuid
from dgl import load_graphs, save_graphs, DGLGraph

AVAILABLE_MODEL = {  # "config_name" : ("module_name", "model_name", "training_process", "requires_features")
    "GAT": {"module_name": "model", "model_name": "GAT", "training_process": "supervised", "requires_features": True},
    "GAT_stochastic": {"module_name": "model", "model_name": "GAT_stochastic", "training_process": "supervised", "requires_features": True},
    "GCN": {"module_name": "model", "model_name": "GCN", "training_process": "supervised", "requires_features": True},
    "GCN_stochastic": {"module_name": "model", "model_name": "GCN_stochastic", "training_process": "supervised", "requires_features": True},
    "GraphSAGE": {"module_name": "model", "model_name": "GraphSAGE", "training_process": "supervised", "requires_features": True},
    "GraphSAGE_stochastic": {"module_name": "model", "model_name": "GraphSAGE_stochastic", "training_process": "supervised", "requires_features": True},
    "log_reg": {"module_name": "sklearn.linear_model", "model_name": "LogisticRegression", "training_process": "supervised", "requires_features": True},
    "deepwalk": {"module_name": "model", "model_name": "DeepWalk", "training_process": "unsupervised", "requires_features": False, "sample_neg_edges_noNN": True},
    "node2vec": {"module_name": "nodevectors", "model_name": "Node2Vec", "training_process": "unsupervised", "requires_features": False, "sample_neg_edges_noNN": True},
    "GAT_LP": {"module_name": "model", "model_name": "GAT_LP", "training_process": "unsupervised", "requires_features": True, "negative_graph": True},
    "GCN_LP": {"module_name": "model", "model_name": "GCN_LP", "training_process": "unsupervised", "requires_features": True, "negative_graph": True},
    "GraphSAGE_LP": {"module_name": "model", "model_name": "GraphSAGE_LP", "training_process": "unsupervised", "requires_features": True, "negative_graph": True},
}

AVAILABLE_DATASET = {  # "dataset name": ("module_name", "dataset_class_name")
    "karate": ("dgl.data", "KarateClubDataset"),
    "coauthor_cs": ("dataset", "RefactorCoauthorCSDataset"),
    "amazon_co_buy_computer": ("dataset", "RefactorAmazonCoBuyComputerDataset"),
    "cora_v2": ("dataset", "RefactorCoraGraphDataset"),
    "citeseer": ("dataset", "RefactorCiteseerGraphDataset"),
    "pubmed": ("dataset", "RefactorPubmedGraphDataset"),
    "gdelt": ("dataset", "RefactorGDELTDataset"),
    "FB15k": ("dataset", "RefactorFB15kDataset"),
    "FB15k-237": ("dataset", "RefactorFB15k237Dataset"),
    "wn18": ("dataset", "RefactorWN18Dataset"),
    "icews18": ("dataset", "RefactorICEWS18Dataset"),
    "ppi": ("dataset", "RefactorPPIDataset"),
    "reddit": ("dataset", "RefactorRedditDataset")
}

LOSS = {
    "binary_cross_entropy_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": F.cross_entropy
}

preprocessor_dict = {
    "sampler": SamplerPreprocessor(),
    "features_sampler": FeaturesSamplerPreprocessor()
}


def set_reproducible(seed=0):
    """
    Sets seeds to have reproducible results.
    """
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)


def instantiate_model(name, params):
    """Instantiantes model.

    Args:
        name (str): name of the model to instantiate.
            Must comply with available keys in AVAILABLE_MODEL at the top.
        params (dictionnary): parameters of the model, mainly
            defined in config files.

    Returns:
        object: instantiated model.
    """
    model_module = AVAILABLE_MODEL[name]["module_name"]
    model_class_name = AVAILABLE_MODEL[name]["model_name"]
    model_class = getattr(import_module(model_module), model_class_name)
    model = model_class(**params)
    return model


def instantiate_dataset(name, params):
    """Instantiantes dataset.

    Args:
        name (str): name of the dataset to instantiate.
            Must comply with available keys in AVAILABLE_DATASET at the top.
        params (dictionnary): parameters of the dataset, mainly
            defined in config files.

    Returns:
        object: instantiated dataset.
    """
    dataset_module, dataset_class_name = AVAILABLE_DATASET[name]
    dataset_class = getattr(import_module(dataset_module), dataset_class_name)
    dataset = dataset_class(**params)
    return dataset


def create_loss(name, params):
    """Instantiates loss.

    Args:
        name (str): name of the loss to instantiate.
            Must comply with available keys in LOSS at the top.
        params (dictionnary): parameters of the loss.

    Returns:
        torch function: instantiated loss.
    """
    loss_fn = LOSS[name](**params)
    return loss_fn


def set_tensors_to_device(data_dict, device):
    """Sends data stored into dictionnary to device.

    Args:
        data_dict (dictionnary): graph data stored into dictionnary.
        device (torch device): working device.

    Returns:
        data_dict: graph data on device.
    """
    for k, v in data_dict.items():
        if hasattr(v, "device") and v.device != device:
            data_dict[k] = v.to(device)
    return data_dict


def generate_save_data_dict_config(config, model_prop_dict):
    """Generates config dictionnary identifying a data_dict save.

    Args:
        config (dict): Experiment config.
        model_prop_dict (dict): Model properties, from AVAILABLE_MODEL.

    Returns:
        dict: Config to use to identify a saved data_dict:
        same dataset & preprocessor, same model graph preprocessing.
    """
    preprocessor_config = deepcopy(config.get("preprocessor", {}))
    model_config = deepcopy(model_prop_dict)
    task = deepcopy(config['experiment'].get("task", {}))
    keys = ["model_name", "module_name"]
    for key in keys:
        model_config.pop(key, None)
    preprocessor_config.update(model_config)
    preprocessor_config.update(task)
    return preprocessor_config


def get_saved_datadict_path_for_config(config, model_prop_dict):
    """If exists, finds path of saved data_dict matching current config:
    same dataset & preprocessor, same model graph preprocessing.


    Args:
        config (dict): Experiment config.
        model_prop_dict (dict): Params of current model, from AVAILABLE_MODEL.

    Returns:
        str: Path to saved data_dict.
    """
    saved_graph_path = os.path.join("data", config["dataset"]["name"],
                                    "saved_graphs")
    current_config = generate_save_data_dict_config(config, model_prop_dict)
    if os.path.exists(saved_graph_path):
        folders = os.listdir(saved_graph_path)
        for folder in folders:
            saved_config = load_config(
                os.path.join(saved_graph_path, folder,
                             "dataset_saved_config.json"))
            if filter_saved_datadict_config(saved_config, config) == filter_saved_datadict_config(current_config, config):
                return os.path.join(saved_graph_path, folder)
    return None

def filter_saved_datadict_config(datadict_config, config):
    """Util function to filter the datadict configs to keep only fields required to compare two configurations

    Args:
        datadict_config (dict): datadict config defining a data dict.
        config (dict): experiment config. 
    """
    if datadict_config.get("name") is None:
        datadict_config.pop("params")
    return datadict_config

def load_saved_data_dict(path):
    """Loads data_dict from path.
    Path must lead to folder containing 'dataset_saved_config.json',
    config of current dataset setting, and all the files to load to the
    data_dict.

    Args:
        path (str): path to the files to load into the data_dict.

    Raises:
        NotImplementedError: If file not supported.

    Returns:
        dict: data_dict, dicitonnary containing necessary data.
    """
    data_files = os.listdir(path)
    data_files.remove("dataset_saved_config.json")
    data_dict = {}
    for data_file in data_files:
        if data_file.endswith(".npy"):
            key, value = data_file.split(".")[0], np.load(
                os.path.join(path, data_file))
            data_dict.update({key: value})
        elif data_file.endswith(".pt"):
            key, value = data_file.split(".")[0], th.load(
                os.path.join(path, data_file))
            data_dict.update({key: value})
        elif data_file.endswith(".bin"):
            graphs, _ = load_graphs(os.path.join(path, data_file))
            key, value = data_file.split(".")[0], graphs[0]
            data_dict.update({key: value})
        elif data_file.endswith(".json"):
            data_dict.update(read_json(os.path.join(path, data_file)))
        else:
            raise NotImplementedError(
                "Type not understood when loading data dict")

    return data_dict


def save_data_dict(config, model_prop_dict, data_dict):
    """Saves data_dict.
    Automatically creates a unique folder, unique to the current dataset config
    (defined from preprocessor used in config & model properties),
    and save each object of the data_dict to a separate file.

    Args:
        config (dict): Experiment config.
        model_prop_dict (dict): Model propertie, from AVAILABLE_MODEL.
        data_dict (dict): data_dict, containing all necessary data to store.

    Raises:
        NotImplementedError: Unsupported format.
    """
    current_config = generate_save_data_dict_config(config, model_prop_dict)
    saved_graph_path = os.path.join("data", config["dataset"]["name"],
                                    "saved_graphs")
    config_folder_name = uuid.uuid4().hex
    output_folder = os.path.join(saved_graph_path, config_folder_name)
    os.makedirs(output_folder)

    save_json(current_config,
              os.path.join(output_folder, "dataset_saved_config.json"))

    int_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            np.save(os.path.join(output_folder, key + ".npy"), value)
        elif isinstance(value, th.Tensor):
            th.save(value, os.path.join(output_folder, key + ".pt"))
        elif isinstance(value, DGLGraph):
            save_graphs(os.path.join(output_folder, key + ".bin"),
                        [value])
        elif isinstance(value, int) or value is None:
            int_dict.update({key: value})
        else:
            raise NotImplementedError("Type: {} cannot be saved - supported format [np.array, th.Tensor, dgl.DGLGraph]".format(type(value)))

    if len(int_dict) != 0:
        save_json(int_dict, os.path.join(output_folder, "int_dict.json"))


def load_data(config, model_prop_dict, logger=None):
    """Loads data to apply models to, including dataset instantiation from DGL,
    preprocessor step if needed and train/val/test masks computation.
    Stores the needed information into a dictionnary.

    Args:
        config (dictionnary): loaded config file.
        model_prop_dict (dictionnary): properties of the model for which data
            is loaded (value of AVAILABLE_MODEL).
        logger (logging Logger, optional): logging system. Defaults to None.

    Returns:
        dictionnary: graph data stored into dictionnary.
    """
    data_dict = {}

    print_info = print if logger is None else logger.info
    print_debug = print if logger is None else logger.debug

    # Load datadict and return if current data params already saved
    if config["experiment"].get("use_saved_datadict", False):
        saved_data_dict_path = get_saved_datadict_path_for_config(
            config, model_prop_dict)
        if saved_data_dict_path is not None:
            print_info("Found saved data_dict matching config, loading datadict..")
            data_dict = load_saved_data_dict(saved_data_dict_path)
            print_info(".. Loaded !")
            return data_dict
        else:
            print_info("No saved data_dict found for this config, continuing..")

    print_debug("Loading data: {}..".format(config["dataset"]["name"]))
    task = config["experiment"]["task"].get(
        "name", None) if "task" in config["experiment"] else None
    dataset_name, dataset_params = config["dataset"]["name"], config[
        "dataset"]["params"]
    dataset = instantiate_dataset(dataset_name, dataset_params)
    graph = dataset[0]  #Note: Not copying, memory safe
    print_debug("Loaded: {}".format(graph))
    if "feat" in graph.ndata and graph.ndata["feat"].ndim == 1:
        graph.ndata["feat"] = th.unsqueeze(graph.ndata["feat"], 1)
    data_dict.update({"graph": graph})

    # Apply data preprocessor
    removed_edges_src, removed_edges_dst = None, None
    if ("preprocessor" in config
            and config["preprocessor"].get("name") is not None):
        print_info("Preprocessing graph for experiments..")
        processor = preprocessor_dict[config["preprocessor"]["name"]]
        graph, removed_edges_src, removed_edges_dst = processor(
            graph, **config["preprocessor"]["params"])
        client_nid = graph.ndata["client"].nonzero().squeeze()
        data_dict.update({"graph": graph, "client_nid": client_nid})
        print_debug(".. Preprocessed: {}".format(graph))
    data_dict.update({
        "removed_edges_src": removed_edges_src,
        "removed_edges_dst": removed_edges_dst
    })

    # Get nodes features
    if model_prop_dict.get("requires_features", True):
        print_info("Getting features..")
        if "feat" in graph.ndata.keys() and config["model"].get(
                "features", True):
            features = graph.ndata["feat"]
        else:
            print_info(".. Setting features to identity")
            features = th.eye(graph.num_nodes())
        data_dict.update({"features": features, "in_feats": features.shape[1]})
        print_info(".. In features: {}".format(data_dict["in_feats"]))
    else:
        data_dict.update({"features": None})

    # Get labels
    if (model_prop_dict.get("training_process") == "supervised"
            or task in ["node_classification", "classic_tabular_model_nc"]):
        print_info("Getting labels..")

        labels = graph.ndata["label"]
        num_classes = len(labels.unique()) if not hasattr(
            dataset, "num_classes") else dataset.num_classes
        data_dict.update({"labels": labels, "n_classes": num_classes})
        print_info(".. Classes: {}".format(data_dict["n_classes"]))
    else:
        data_dict.update({"labels": None})

    # Change labels values if do not start at 0: pytorch requirement
    if (data_dict["labels"] is not None) and\
            (not all(data_dict["labels"].unique() == th.tensor(range(data_dict["n_classes"])))):
        print_info('Remapping labels..')
        labels = data_dict["labels"]
        unique_labels = labels.unique()
        mapping = {
            old.item(): new
            for old, new in zip(unique_labels, range(data_dict["n_classes"]))
        }
        data_dict["labels"] = th.tensor([mapping[x.item()] for x in labels])

    # Get training nodes masks.
    if (model_prop_dict.get("training_process") == "supervised") or\
            (task in ["node_classification", "classic_tabular_model_nc"]):
        train_mask, val_mask, test_mask = graph.ndata[
            "train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
        train_nid, val_nid, test_nid = th.nonzero(train_mask).squeeze(
        ), th.nonzero(val_mask).squeeze(), th.nonzero(test_mask).squeeze()
        data_dict.update({
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "train_nid": train_nid,
            "val_nid": val_nid,
            "test_nid": test_nid
        })
        if model_prop_dict.get("sample_neg_edges"):
            data_dict.update({"adj": graph.adjacency_matrix().to_dense()})

    # Get training edges masks.
    if model_prop_dict.get("training_process") == "unsupervised":
        if model_prop_dict.get("negative_graph"):
            print_info("Getting training mask")
            if "train_mask" not in graph.edata.keys():
                print_info("Edge masks not available, creating..")
                train_eid, val_eid, test_eid = create_train_val_test_split_mask_edges(
                    graph,
                    train_size=0.85,
                    val_size=0.05,
                    test_size=0.1,
                    respect_class_proportion=False)
                train_graph = graph.edge_subgraph(train_eid,
                                                  preserve_nodes=True)
                val_graph = graph.edge_subgraph(val_eid, preserve_nodes=True)
                test_graph = graph.edge_subgraph(test_eid, preserve_nodes=True)
            else:
                train_graph = graph.edge_subgraph(graph.edata["train_mask"])
                val_graph = graph.edge_subgraph(graph.edata["val_mask"])
                test_graph = graph.edge_subgraph(graph.edata["test_mask"])
            data_dict.update({
                "train_graph": train_graph,
                "val_graph": val_graph,
                "test_graph": test_graph
            })
        elif (model_prop_dict.get("sample_neg_edges") and task == "link_prediction") or\
                (model_prop_dict.get("sample_neg_edges_noNN") and task == "link_prediction"):
            print_info(
                "Preprocessing graph before training: edges masks creation..")
            graph, val_pos_edges, test_pos_edges, val_neg_edges, test_neg_edges = unsup_graph_preprocessor(
                graph)
            data_dict.update({"adj": graph.adjacency_matrix().to_dense()})
            data_dict.update({
                "graph": graph,
                "val_pos_edges": val_pos_edges,
                "test_pos_edges": test_pos_edges,
                "val_neg_edges": val_neg_edges,
                "test_neg_edges": test_neg_edges
            })
            all_test_edges = np.concatenate(
                (data_dict["test_pos_edges"], data_dict["test_neg_edges"]))
            data_dict.update({"all_test_edges": all_test_edges})
            print_debug("..Preprocessed: {}".format(graph))
        else:
            pass

    if config["experiment"].get("use_saved_datadict", False):
        print_info(
            "Config argument 'use_saved_datadict' set to 'True', without existing save, saving this data_dict.."
        )
        save_data_dict(config, model_prop_dict, data_dict)

    return data_dict


def reduce_graph_to_maxsubgraph(edges_list, max_subgraph, mapping_nodes):
    """Reduces graph from edges_list to its max_subgraph.

    Args:
        edges_list (list[tuple]): source and destination nodes indexes
            to translate into max subgraph indexes.
        max_subgraph (list[tuple]): source and destination nodes indexes
            included in max subgraph.
        mapping_nodes (dictionnary): dictionnary mapping nodes indexes
            between main graph and max subgraph.

    Returns:
        list[tuple]: source and destination nodes with max subgraph indexes.
    """
    G = nx.Graph()
    G.add_edges_from(edges_list)
    G = G.subgraph(max_subgraph)
    G = nx.relabel_nodes(G, mapping_nodes)
    return np.array([e for e in G.edges()])


def reduce_to_max_subgraph(task, data_dict, logger=None):
    """Reduces graph and train/val/test masks to nodes and edges available
    in max subgraph. Inplace.

    Args:
        task (str): defines task ("link_prediction", "node_classification",
        "classic_tabular_model_nc").
        data_dict (dictionnary): graph data stored into dictionnary.
        logger (logging Logger, optional): logging system. Defaults to None.
    """
    print_info = print if logger is None else logger.info
    print_info("Restriction to the bigger subgraph..")
    nx_graph = data_dict["nx_graph"]
    max_subgraph = sorted(nx.connected_components(nx_graph),
                          key=len,
                          reverse=True)[0]
    nx_graph = nx_graph.subgraph(max_subgraph)
    mapping_nodes = dict(
        zip(list(nx_graph.nodes()), list(range(nx_graph.number_of_nodes()))))

    nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    if task == "node_classification":
        labels = data_dict["labels"]
        train_mask, val_mask, test_mask = data_dict["train_mask"], data_dict[
            "val_mask"], data_dict["test_mask"]
        labels = labels[list(max_subgraph)]
        train_mask, val_mask, test_mask = train_mask[list(
            max_subgraph)], val_mask[list(max_subgraph)], test_mask[list(
                max_subgraph)]
        data_dict.update({
            "nx_graph": nx_graph,
            "labels": labels,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask
        })

    elif task == 'link_prediction':
        test_pos_edges = data_dict["test_pos_edges"]
        test_neg_edges = data_dict["test_neg_edges"]
        test_pos_edges = reduce_graph_to_maxsubgraph(test_pos_edges,
                                                     max_subgraph,
                                                     mapping_nodes)
        test_neg_edges = reduce_graph_to_maxsubgraph(test_neg_edges,
                                                     max_subgraph,
                                                     mapping_nodes)
        print_info(
            "Stats - Test positives edges: {}, Test negative edges: {}".format(
                test_pos_edges.shape[0], test_neg_edges.shape[0]))
        if test_pos_edges.shape[0] == 0 or test_neg_edges.shape[0] == 0:
            return
        nb_elt_test = min(test_pos_edges.shape[0], test_neg_edges.shape[0])
        test_pos_edges = test_pos_edges[:nb_elt_test]
        test_neg_edges = test_neg_edges[:nb_elt_test]
        all_test_edges = np.concatenate((test_pos_edges, test_neg_edges))
        data_dict.update({
            "nx_graph": nx_graph,
            "test_pos_edges": test_pos_edges,
            "test_neg_edges": test_neg_edges,
            "all_test_edges": all_test_edges
        })
