#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import argparse
import logging
import os
import torch as th
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


from utils.graphs_utils import construct_negative_graph
from utils.experiments_utils import (ExperimentManager, load_config,
                                     merge_metrics, print_dict, save_json)
from utils.main_utils import (AVAILABLE_MODEL, instantiate_model,
                              load_data, set_reproducible,
                              set_tensors_to_device)
from utils.train_utils import EarlyStopping


def train(model, exp_manager, config, device, tag_rep="", logger=None, **data_dict):
    train_graph, val_graph, features, labels = data_dict["train_graph"], data_dict["val_graph"], data_dict["features"], data_dict.get("labels", None)
    removed_edges = (data_dict["removed_edges_src"], data_dict["removed_edges_dst"])

    optimizer = th.optim.Adam(model.parameters(), **config["training"]["optimizer"])
    loss_fn = F.binary_cross_entropy_with_logits

    early_stopping = EarlyStopping(**config["training"]["early_stopping"])

    k = config["experiment"]["task"]["params"]["k"]
    threshold_proba = config["experiment"]["task"]["params"]["threshold_proba"]

    if config["training"]["loss"]["use_weights"]:
        bal_ratio = 1/(k+1)
        weights = th.cat((th.ones((train_graph.num_edges(), 1))*(1-bal_ratio), th.ones((train_graph.num_edges()*k, 1))*bal_ratio)).to(device)
    else:
        weights = None

    metrics = None
    for epoch in range(1, config["training"]["n_epochs"] + 1):
        negative_graph = construct_negative_graph(train_graph, k, device)
        model, loss = train_it_negative_graph(model, train_graph, negative_graph, features, labels, loss_fn, weights, optimizer, exp_manager, epoch, device)

        eval_metrics = {"epoch": epoch, "loss": loss.item()}
        eval_metrics.update(evaluate(model, features, val_graph, k, threshold_proba, device, removed_edges))
        print_dict(eval_metrics)
        metrics = merge_metrics(metrics, eval_metrics)

        if epoch % config["experiment"]["ckpt_save_interval"] == 0 and not config["experiment"]["ghost"]:
            logger.info("Saving model.. ")
            th.save(model.state_dict(), os.path.join(exp_manager.output_path, "saved_model", tag_rep, config["model"]["name"] + "_" + str(epoch) + ".pth"))

        early_stopping_metric = "accuracy_lp"
        if early_stopping(-metrics[early_stopping_metric][-1]):
            logger.debug("Loss not decreasing since {} epochs, stopping..".format(early_stopping.patience))
            break

    return model, metrics


def train_it_negative_graph(model, train_graph, negative_graph, features, labels, loss_fn, weights, optimizer, exp_manager, epoch, device):
    model.train()
    pos_score = model(train_graph, features)
    neg_score = model(negative_graph, features)

    train_labels = th.cat((th.ones(pos_score.size()), th.zeros(neg_score.size())))
    train_labels = train_labels.to(device) if "cuda" in device.type else train_labels

    loss = loss_fn(th.cat([pos_score, neg_score]), train_labels, weight=weights)

    optimizer.zero_grad()
    loss.backward()
    exp_manager.monitor_epoch(model, labels, epoch)
    optimizer.step()
    return model, loss


def add_evaluation(model, previous_metrics, graph_name, config, device, logger=None, tag="", verbose=1, *args, **kwargs):
    graph, features = kwargs[graph_name], kwargs["features"]
    threshold_proba = config["experiment"]["task"]["params"]["threshold_proba"]
    removed_edges = (kwargs["removed_edges_src"], kwargs["removed_edges_dst"])
    metrics = evaluate(model, features, graph, 1, threshold_proba, device, removed_edges)
    if verbose > 0:
        print_dict(metrics, logger=logger)
    return merge_metrics(previous_metrics, metrics, tag=tag)


def evaluate(model, features, graph, k, threshold_proba, device, removed_edges):
    negative_graph = construct_negative_graph(graph, k, device)
    model.eval()

    nb_removed_edges = 0
    neg_mask = th.tensor([True]*negative_graph.number_of_edges())
    if removed_edges != (None, None):
        # Do not evaluate on removed edges when graph preprocessed
        rem_edges_mask = negative_graph.has_edges_between(removed_edges[0], removed_edges[1])
        neg_rem_eids = negative_graph.edge_ids(removed_edges[0][rem_edges_mask], removed_edges[1][rem_edges_mask])
        neg_mask[neg_rem_eids] = False
        nb_removed_edges = (~neg_mask).sum().item()

    with th.no_grad():
        pos_score = model(graph, features)
        all_neg_score = model(negative_graph, features)
        neg_score = all_neg_score[neg_mask]
        pos_labels = th.where(th.sigmoid(pos_score) > threshold_proba, th.ones_like(pos_score, dtype=th.int32).to(device), th.zeros_like(pos_score, dtype=th.int32).to(device)) 
        neg_labels = th.where(th.sigmoid(neg_score) > threshold_proba, th.ones_like(neg_score, dtype=th.int32).to(device), th.zeros_like(neg_score, dtype=th.int32).to(device))

        _pred_score = th.cat([pos_score, neg_score]).cpu()
        _labels = th.cat([th.ones_like(pos_score, dtype=th.int32), th.zeros_like(neg_score, dtype=th.int32)]).cpu()
        auc = roc_auc_score(_labels, _pred_score)
        ap = average_precision_score(_labels, _pred_score)

        all_neg = neg_score.size()[0]
        all_pos = pos_score.size()[0]
        tp = (pos_labels == 1).sum().item()
        fp = (neg_labels == 1).sum().item()
        tn = all_neg - fp

        precision = tp / (tp + fp + 1e-6)
        recall = tp/all_pos
        accuracy = (tp + tn) / (all_neg + all_pos)
        f1 = 2*recall*precision/(recall + precision + 1e-6)

        if nb_removed_edges != 0:
            neg_score_rem_edges = all_neg_score[~neg_mask]
            neg_labels_rem_edges = th.where(th.sigmoid(neg_score_rem_edges) > threshold_proba, th.ones_like(neg_score_rem_edges , dtype=th.int32).to(device), th.zeros_like(neg_score_rem_edges , dtype=th.int32).to(device))
            accuracy_removed_edges = (neg_labels_rem_edges == 1).sum().item()/len(neg_labels_rem_edges)
        else:
            accuracy_removed_edges = 0

    return {"precision_lp": precision, "recall_lp": recall, "accuracy_lp": accuracy, "f1_lp": f1, "auc_lp": auc, "ap_lp": ap, "nb_removed_edges": nb_removed_edges, "accuracy_removed_edges": accuracy_removed_edges}


def set_model(config, data_dict, device):
    model_name, model_params = config["model"]["name"], config["model"]["params"]
    train_graph = data_dict["train_graph"].to(device) if "cuda" in device.type else data_dict["train_graph"]
    model_params.update({"g": train_graph, "in_dim": data_dict["in_feats"]})
    model = instantiate_model(model_name, model_params)
    return model


def main(data_dict, args, config, exp_manager, logger):
    logger.debug("Starting main..")
    device = th.device("cpu" if args.gpu<0 else "cuda: "+str(args.gpu))

    all_metrics = []
    for rep in range(1, config["experiment"]["repetitions"]+1):
        logger.info("***************** Starting repetition: {} ***************".format(rep))
        tag_rep = "rep_{}".format(rep) if config["experiment"]["repetitions"] > 1 else ""

        logger.debug("Instantiating model {}..".format(config["model"]["name"]))
        model = set_model(config, data_dict, device)
        logger.info(".. Model: {}".format(model))

        if args.gpu >= 0:
            logger.debug("Setting data to: {}".format(device))
            data_dict = set_tensors_to_device(data_dict, device)
            model = model.to(device)

        logger.debug("Train model..")
        model, metrics = train(model, exp_manager, config, device, tag_rep, logger, **data_dict)

        logger.info("Computing test metrics..")
        metrics = add_evaluation(model, metrics, "test_graph", config, device, logger=logger, tag="_test", **data_dict)

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
    parser.add_argument("--config", type=str, default="gcn_lp", help="Name of the configuration file to run experiment from, without .json extension")
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
            main(data_dict, args, config, exp_manager, logger)
        except Exception as e:
            logger.exception("Exception occurred during main task : {}".format(e))
