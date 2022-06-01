#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import argparse
import logging
import os
import numpy as np
import torch as th
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.graphs_utils import sample_train_edges
from utils.experiments_utils import (load_config, ExperimentManager,
                                     save_json)
from utils.experiments_utils import merge_metrics, print_dict
from utils.main_utils import (AVAILABLE_MODEL,
                              instantiate_model, load_data, set_reproducible,
                              reduce_to_max_subgraph)

os.environ['PYTHONHASHSEED'] = '0'


def create_sets_evaluation(task, data_dict, removed_edges=None, logger=None):
    if task == "node_classification":
        embedding, labels = data_dict["embedding"], data_dict["labels"]
        train_mask, test_mask = data_dict["train_mask"], data_dict["test_mask"]
        X_train = embedding[train_mask]
        labels_train = labels[train_mask]
        X_test = embedding[test_mask]
        labels_test = labels[test_mask]

    elif task == "link_prediction":
        graph, embedding = data_dict["nx_graph"], data_dict["embedding"]
        all_test_edges, test_pos_edges, test_neg_edges = data_dict["all_test_edges"], data_dict["test_pos_edges"], data_dict["test_neg_edges"]

        pos_train_edges = np.array([e for e in graph.edges()])
        neg_train_edges = sample_train_edges(graph, pos_train_edges, all_test_edges)

        if removed_edges != (None, None):
            # Drop removed edges when graph preprocesssed from test and train
            # negative sets
            dgl_graph = data_dict['graph']

            train_mask = dgl_graph.has_edges_between(neg_train_edges[:, 0], neg_train_edges[:, 1]) +\
                dgl_graph.has_edges_between(neg_train_edges[:, 1], neg_train_edges[:, 0])
            neg_train_edges = neg_train_edges[~train_mask]

            test_neg_mask = dgl_graph.has_edges_between(test_neg_edges[:, 0], test_neg_edges[:, 1]) +\
                dgl_graph.has_edges_between(test_neg_edges[:, 1], test_neg_edges[:, 0])
            all_test_mask = th.cat((th.tensor([False]*len(test_pos_edges)), test_neg_mask))
            all_test_edges = all_test_edges[~all_test_mask]
            print("{} negative edges are removed from train set and {} from test set (positive edges before sampling)".format(train_mask.sum().item(), test_neg_mask.sum().item()))
            del dgl_graph

        all_train_edges = np.concatenate((pos_train_edges, neg_train_edges))
        logger.debug("Stats - Train positives edges: {}, Train negative edges: {}".format(pos_train_edges.shape[0], neg_train_edges.shape[0]))

        X_train = np.concatenate((embedding[all_train_edges[:, 0]], embedding[all_train_edges[:, 1]]), axis=1)
        labels_train = np.concatenate((np.ones(pos_train_edges.shape[0]), np.zeros(neg_train_edges.shape[0])))

        X_test = np.concatenate((embedding[all_test_edges[:, 0]], embedding[all_test_edges[:, 1]]), axis=1)
        labels_test = np.concatenate((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))

    else:
        raise NotImplementedError

    data_dict.update({"X_train": X_train, "X_test": X_test, "labels_train": labels_train, "labels_test": labels_test})


def train_evaluation(config, X_train, labels_train, logger):
    logger.debug("Instantiating evaluation model {}..".format(config["evaluation"]["model"]["name"]))
    eval_model_name, eval_model_params = config["evaluation"]["model"]["name"], config["evaluation"]["model"]["params"]
    eval_model = instantiate_model(eval_model_name, eval_model_params)
    logger.info(".. Evaluation model: {}".format(eval_model))

    eval_model.fit(X_train, labels_train)

    return eval_model


def add_evaluation(config, metrics, data_dict, removed_edges, logger, verbose=1):
    create_sets_evaluation(config["experiment"]["task"]["name"], data_dict, removed_edges, logger=logger)
    X_train, X_test, labels_train, labels_test = data_dict["X_train"], data_dict["X_test"], data_dict["labels_train"], data_dict["labels_test"]

    eval_model = train_evaluation(config, X_train, labels_train, logger)

    logger.info("Computing train metrics..")
    eval_metrics = evaluate(eval_model, X_train, labels_train)
    if verbose > 0:
        print_dict(eval_metrics, logger=logger)
    metrics = merge_metrics(metrics, eval_metrics)

    logger.info("Computing test metrics..")
    test_metrics = evaluate(eval_model, X_test, labels_test)
    if verbose > 0:
        print_dict(test_metrics, logger=logger)
    metrics = merge_metrics(metrics, test_metrics, tag='_test')

    return metrics


def evaluate(eval_model, embedding, labels):
    _proba_pos_class = eval_model.predict_proba(embedding)[:, eval_model.classes_ == 1].reshape((-1))
    auc = roc_auc_score(labels, _proba_pos_class)
    ap = average_precision_score(labels, _proba_pos_class)

    _labels_pred = eval_model.predict(embedding)
    tp = np.logical_and(_labels_pred, labels).sum()

    precision = (tp / _labels_pred.sum()).item()
    recall = (tp / labels.sum()).item()
    accuracy = np.sum(_labels_pred == labels).item() * 1.0/len(labels)
    f1 = 2*recall*precision/(recall+precision + 1e-6)

    return {"precision_lp": precision, "recall_lp": recall, "accuracy_lp": accuracy, "f1_lp": f1, "auc_lp": auc, "ap_lp": ap}


def add_node_class_evaluation(config, metrics, data_dict, logger, verbose=1):
    list_metrics = ["accuracy", "f1", "recall", "precision"]

    create_sets_evaluation(config["experiment"]["task"]["name"], data_dict)
    X_train, X_test, labels_train, labels_test = data_dict["X_train"], data_dict["X_test"], data_dict["labels_train"], data_dict["labels_test"]

    eval_model = train_evaluation(config, X_train, labels_train, logger)

    logger.info("Computing train metrics..")
    eval_metrics = evaluate_node_classif(eval_model, X_train, labels_train.numpy(), list_metrics, logger, verbose)
    metrics = merge_metrics(metrics, eval_metrics, tag="")

    logger.info("Computing test metrics for..")
    test_metrics = evaluate_node_classif(eval_model, X_test, labels_test.numpy(), list_metrics, logger, verbose)
    metrics = merge_metrics(metrics, test_metrics, tag="_test")

    return metrics


def evaluate_node_classif(eval_model, embedding, labels, metrics_to_cpt, logger, verbose):
    _labels_pred = eval_model.predict(embedding)
    dict_metrics = {}

    accuracy = np.sum(_labels_pred == labels).item() * 1.0/len(labels)
    dict_metrics["accuracy"] = accuracy

    unique_labels = np.unique(labels)
    _labels_pred = label_binarize(_labels_pred, classes=unique_labels)
    labels = label_binarize(labels, classes=unique_labels)
    iterable = range(len(unique_labels)) if len(unique_labels) > 2 else range(len(unique_labels)-1)

    precision_class = []
    recall_class = []
    f1_class = []

    for i in iterable:
        tp = np.logical_and(_labels_pred[:, i], labels[:, i]).sum()

        precision = (tp / _labels_pred[:, i].sum()).item()
        precision_class.append(precision)

        recall = (tp / labels[:, i].sum()).item()
        recall_class.append(recall)

        f1 = 2*recall*precision/(recall+precision + 1e-6)
        f1_class.append(f1)

    if "precision" in metrics_to_cpt:
        dict_metrics["precision_macro"] = sum(precision_class)/len(precision_class)
    if "recall" in metrics_to_cpt:
        dict_metrics["recall_macro"] = sum(recall_class)/len(recall_class)
    if "f1" in metrics_to_cpt:
        dict_metrics["f1_macro"] = sum(f1_class)/len(f1_class)

    if verbose > 0:
        print_dict(dict_metrics, logger=logger)

    if "precision" in metrics_to_cpt:
        dict_metrics["precision"] = precision_class
    if "recall" in metrics_to_cpt:
        dict_metrics["recall"] = recall_class
    if "f1" in metrics_to_cpt:
        dict_metrics["f1"] = f1_class

    return dict_metrics


def main(data_dict, args, config, exp_manager, logger):
    logger.debug("Starting main..")
    verbose = args.verbose
    task = config["experiment"]["task"]["name"]

    graph = data_dict["graph"]
    nx_graph = graph.to_networkx().to_undirected()
    data_dict.update({"nx_graph": nx_graph})
    if config["experiment"]["only_max_subgraph"]:
        reduce_to_max_subgraph(task, data_dict, logger)

    all_metrics = []
    for rep in range(1, config["experiment"]["repetitions"]+1):
        logger.info("***************** Starting repetition: {} ***************".format(rep))
        tag_rep = "rep_{}".format(rep) if config["experiment"]["repetitions"] > 1 else ""

        logger.debug("Instantiating model {}..".format(config["model"]["name"]))
        model_name, model_params = config["model"]["name"], config["model"]["params"]
        model = instantiate_model(model_name, model_params)
        logger.info(".. Model: {}".format(model_name))

        logger.debug("Train embedding model..")
        nx_graph = data_dict["nx_graph"]

        if hasattr(model, "get_embedding"):
            model.fit(nx_graph)
            embedding = model.get_embedding()
        else:
            embedding = model.fit_transform(nx_graph)
        data_dict.update({"embedding": embedding})

        metrics = None
        if task == "link_prediction":
            removed_edges = (data_dict["removed_edges_src"], data_dict["removed_edges_dst"])
            metrics = add_evaluation(config, metrics, data_dict, removed_edges, logger, verbose)

        if task == "node_classification":
            logger.info("Computing {} metrics..".format(task))
            metrics = add_node_class_evaluation(config, metrics, data_dict, logger, verbose)

        if not config["experiment"]["ghost"]:
            logger.info("Saving results to {}".format(os.path.join(exp_manager.output_path, "results", tag_rep, "metrics.json")))
            save_json(metrics, os.path.join(exp_manager.output_path, "results", tag_rep, "metrics.json"))
            all_metrics.append(metrics)

    save_json(all_metrics, os.path.join(exp_manager.output_path, "all_metrics.json"))
    logger.debug("End main..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", type=int, default=2, help="Logging level and other information. Higher value returns more specific information")
    parser.add_argument("--config", type=str, default="deepwalk", help="Name of the configuration file to run experiment from, without .json extension")
    parser.add_argument("--tag", type=str, default="", help="Specify tag to recognize experiment easily")
    parser.add_argument("--gpu", type=int, default=0, help="cpu: <0, gpu: >=0 ")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset name")
    parser.add_argument("--rep", default=None, type=int, help="Number of repetitions of the model training & evaluation")
    parser.add_argument("--task", default=None, type=str, help="Downstream task to use embeddings for. Either 'link_prediction' or 'node_classification'", choices=["link_prediction", "node_classification"])
    parser.add_argument("--preprocessor", default=None, type=str, help="Specify preprocessor. Either 'sampler', or 'features_sampler'", choices=["sampler", "features_sampler"])
    parser.add_argument("--sampling_ratio", default=None, type=float, help="Sampling ratio to identify the portion of client nodes")
    parser.add_argument("--features_sampling_ratio", default=None, type=float, help="Sampling ratio for nodes features deterioration")
    parser.add_argument("--graph_rep", default=1, type=int, help="Number of repetitition of the experiment (graph preprocessing + multiple model trainings & evaluations)")

    args = parser.parse_args()

    set_reproducible()

    for rep in range(1, args.graph_rep+1): 

        config = load_config(os.path.join("configs", args.config + ".json"))

        exp_manager = ExperimentManager(args, config)

        logger = logging.getLogger(__name__)
        exp_manager.set_logger(logger)

        try:
            data_dict = load_data(config, AVAILABLE_MODEL[config["model"]["name"]], logger=logger)
            main(data_dict, args, config, exp_manager, logger)
        except Exception as e:
            logger.exception("Exception occurred during main task : {}".format(e))
