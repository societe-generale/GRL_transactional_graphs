#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import argparse
import logging
import os
import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

from utils.experiments_utils import (ExperimentManager, load_config,
                                     merge_metrics, print_dict,
                                     save_json)
from utils.main_utils import (AVAILABLE_MODEL, create_loss,
                              instantiate_model, load_data,
                              set_reproducible, set_tensors_to_device)
from utils.train_utils import EarlyStopping
import sklearn
import tqdm


def train(model, exp_manager, config, tag_rep="", logger=None, *args, **kwargs):
    features, labels, train_nid, val_nid = kwargs["features"], kwargs["labels"], kwargs["train_nid"], kwargs["val_nid"]
    optimizer = th.optim.Adam(model.parameters(), **config["training"]["optimizer"])
    loss_fn = F.cross_entropy

    early_stopping = EarlyStopping(**config["training"]["early_stopping"])

    metrics = None
    for epoch in range(1, config["training"]["n_epochs"] + 1):
        model, logits, loss = train_it_supervised(model,
                                                  features,
                                                  labels,
                                                  loss_fn,
                                                  train_nid,
                                                  optimizer,
                                                  exp_manager,
                                                  epoch)

        val_loss = loss_fn(logits[val_nid], labels[val_nid])
        eval_metrics = {
            "epoch": epoch,
            "loss": loss.item(),
            "val_loss": val_loss.item()
            }
        print_dict(eval_metrics)
        eval_metrics.update(evaluate(model, features, labels, val_nid, logger, 1))
        metrics = merge_metrics(metrics, eval_metrics)

        if epoch % config["experiment"]["ckpt_save_interval"] == 0 and not config["experiment"]["ghost"]:
            logger.info("Saving model.. ")
            th.save(model.state_dict(), os.path.join(exp_manager.output_path, "saved_model", tag_rep, config["model"]["name"] + "_" + str(epoch) + ".pth"))

        early_stopping_metric = "val_loss"
        if early_stopping(metrics.get(early_stopping_metric, [np.inf])[-1]):
            logger.debug("Loss not decreasing since {} epochs, stopping..".format(early_stopping.patience))
            break

    return model, metrics


def train_it_supervised(model, features, labels, loss_fn, train_nid, optimizer, exp_manager, epoch):
    model.train()
    logits = model(features)
    loss = loss_fn(logits[train_nid], labels[train_nid])

    optimizer.zero_grad()
    loss.backward()
    exp_manager.monitor_epoch(model, labels, epoch)
    optimizer.step()
    return model, logits, loss


def train_stochastic(model, exp_manager, config, tag_rep="", logger=None, device="cpu", *args, **kwargs):
    graph, train_nid, val_nid = kwargs["graph"], kwargs["train_nid"], kwargs["val_nid"]
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config["model"]["params"]["num_layers"]) # autres que reddit
    sampler = dgl.dataloading.MultiLayerNeighborSampler([20]*config["model"]["params"]["num_layers"]) # reddit
    dataloader = dgl.dataloading.NodeDataLoader(graph, train_nid, sampler, device=device, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False, num_workers=8)

    optimizer = th.optim.Adam(model.parameters(), **config["training"]["optimizer"])
    loss_fn = F.cross_entropy

    early_stopping = EarlyStopping(**config["training"]["early_stopping"])

    metrics = None
    for epoch in range(1, config["training"]["n_epochs"] + 1):
        model.train()

        with tqdm.tqdm(dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):

                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]

                logits = model(mfgs, inputs)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                exp_manager.monitor_epoch(model, labels, epoch)

                acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), logits.argmax(1).detach().cpu().numpy())

                tq.set_postfix({"loss": "%.3f" % loss.item(), "acc": "%.3f" % acc}, refresh=False)

        eval_metrics = {"epoch": epoch, "loss": loss.item()}
        print_dict(eval_metrics)
        if epoch % config["experiment"].get("val_metrics_interval", 1) == 0:
            eval_metrics.update(evaluate_stochastic(model, None, labels, val_nid, logger, verbose=1, graph=graph, config=config, device=device))
        else:
            eval_metrics = {}
        metrics = merge_metrics(metrics, eval_metrics)


        if epoch % config["experiment"]["ckpt_save_interval"]  == 0 and not config["experiment"]["ghost"]:
            logger.info("Saving model.. ")
            th.save(model.state_dict(), os.path.join(exp_manager.output_path, "saved_model", tag_rep, config["model"]["name"] + "_" + str(epoch) + ".pth"))

        early_stopping_metric = "val_loss"
        if early_stopping(metrics.get(early_stopping_metric, [np.inf])[-1]):
            logger.debug("Loss not decreasing since {} epochs, stopping..".format(early_stopping.patience))
            break

    return model, metrics


def add_evaluation(model, previous_metrics, ids_name, config, device, logger=None, tag="", verbose=1, *args, **kwargs):
    graph, features, labels, ids = kwargs["graph"], kwargs["features"], kwargs["labels"], kwargs[ids_name]
    evaluate_fn = evaluate_stochastic if config["model"]["name"].endswith("stochastic") else evaluate
    metrics = evaluate_fn(model, features, labels, ids, logger, verbose, graph=graph, device=device, config=config)
    return merge_metrics(previous_metrics, metrics, tag=tag)


def evaluate(model, features, labels, nid, logger, verbose, graph=None, device=None, config=None, *args):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[nid]
        labels = labels[nid].cpu().numpy()
        _, _labels_pred = th.max(logits, dim=1)
        _labels_pred = _labels_pred.cpu().numpy()

        dict_metrics = {}

        accuracy = np.sum(_labels_pred == labels)/len(labels)
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

        dict_metrics["precision_macro"] = sum(precision_class)/len(precision_class)
        dict_metrics["recall_macro"] = sum(recall_class)/len(recall_class)
        dict_metrics["f1_macro"] = sum(f1_class)/len(f1_class)

        if verbose > 0:
            print_dict(dict_metrics, logger=logger)
        dict_metrics["precision"] = precision_class
        dict_metrics["recall"] = recall_class
        dict_metrics["f1"] = f1_class
    return dict_metrics


def evaluate_stochastic(model, features, labels, nid, logger, verbose, graph, device, config):
    model.eval()
    with th.no_grad():
        sampler = dgl.dataloading.MultiLayerNeighborSampler([50]*config["model"]["params"]["num_layers"])
        dataloader = dgl.dataloading.NodeDataLoader(graph, nid, sampler, device=device, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=False, num_workers=8)

        logits_list, labels_list = [], []
        for setp, (input_nodes, output_nodes, mfgs) in enumerate(dataloader):

            inputs = mfgs[0].srcdata["feat"]
            labels = mfgs[-1].dstdata["label"]

            logits = model(mfgs, inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = th.cat(logits_list, dim=0)
    labels = th.cat(labels_list, dim=0)
    loss_fn = F.cross_entropy
    val_loss = loss_fn(logits, labels)
    labels = labels.cpu().numpy()
    _, _labels_pred = th.max(logits, dim=1)
    _labels_pred = _labels_pred.cpu().numpy()

    dict_metrics = {"loss": val_loss.item()}

    accuracy = np.sum(_labels_pred == labels)/len(labels)
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

    dict_metrics["precision_macro"] = sum(precision_class)/len(precision_class)
    dict_metrics["recall_macro"] = sum(recall_class)/len(recall_class)
    dict_metrics["f1_macro"] = sum(f1_class)/len(f1_class)

    if verbose > 0:
        print_dict(dict_metrics, logger=logger)
    dict_metrics["precision"] = precision_class
    dict_metrics["recall"] = recall_class
    dict_metrics["f1"] = f1_class
    return dict_metrics


def set_model(config, data_dict, device):
    model_name, model_params = config["model"]["name"], config["model"]["params"]
    graph = data_dict["graph"].to(device) if "cuda" in device.type else data_dict["graph"]
    model_params.update({"g": graph, "in_dim": data_dict["in_feats"], "num_classes": data_dict["n_classes"]})
    model = instantiate_model(model_name, model_params)
    return model


def main(data_dict, args, config, exp_manager, logger):
    logger.debug("Starting main..")
    verbose = args.verbose
    device = th.device("cpu" if args.gpu < 0 else "cuda: "+str(args.gpu))
    stochastic_training = True if config["model"]["name"].endswith("stochastic") else False

    all_metrics = []
    for rep in range(1, config["experiment"]["repetitions"]+1):
        logger.info("***************** Starting repetition: {} ***************".format(rep))
        tag_rep = "rep_{}".format(rep) if config["experiment"]["repetitions"] > 1 else ""

        logger.debug("Instantiating model {}..".format(config["model"]["name"]))
        model = set_model(config, data_dict, device)
        logger.info(".. Model: {}".format(model))

        if args.gpu >= 0:
            logger.debug("Setting model and data to: {}".format(device))
            if not stochastic_training:
                data_dict = set_tensors_to_device(data_dict, device)
            model = model.to(device)

        logger.debug("Train model..")
        train_fn = train_stochastic if stochastic_training else train
        model, metrics = train_fn(model, exp_manager, config, tag_rep, logger=logger, device=device, **data_dict)

        logger.info("Computing test metrics..")
        metrics = add_evaluation(model, metrics, "test_nid", config, device, logger=logger, tag="_test", verbose=verbose, **data_dict)

        if not config["experiment"]["ghost"]:
            logger.info("Saving final model")
            th.save(model.state_dict(), os.path.join(exp_manager.output_path, "saved_model", tag_rep, config["model"]["name"] + "_" + str(metrics["epoch"][-1]) + "_final" + ".pth"))

            logger.info("Saving results to {}".format(os.path.join(exp_manager.output_path, "results", tag_rep, "metrics.json")))
            save_json(metrics, os.path.join(exp_manager.output_path, "results", tag_rep, "metrics.json"))

            all_metrics.append(metrics)

    save_json(all_metrics, os.path.join(exp_manager.output_path, "all_metrics.json"))
    logger.debug("End main..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", type=int, default=2, help="Logging level and other information. Higher value returns more specific information")
    parser.add_argument("--config", type=str, default="gcn", help="Name of the configuration file to run experiment from, without .json extension")
    parser.add_argument("--tag", type=str, default="", help="Specify tag to recognize experiment easily")
    parser.add_argument("--gpu", type=int, default=0, help="cpu: <0, gpu: >=0 ")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset name")
    parser.add_argument("--rep", default=None, type=int, help="Number of repetitions of the model training & evaluation")
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
            main(data_dict, args, config, exp_manager, logger=logger)
        except Exception as e:
            logger.exception("Exception occurred during main task : {}".format(e))
