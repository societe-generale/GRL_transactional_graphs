#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import json
import logging
import math
import os
import re
import shutil
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
import numpy as np
import torch as th


def load_config(cfg_path):
    """
    Loads config file in .json format.
    """
    assert os.path.exists(cfg_path)
    assert os.path.isfile(cfg_path)
    assert os.path.splitext(cfg_path)[-1] == ".json"
    with open(cfg_path, "r") as f:
        config = json.load(f)
    assert isinstance(config, dict)

    return config


def save_json(file, fp):
    """
    Saves the object file in fp (path to json format file).
    """
    with open(fp, "w") as fp:
        json.dump(file, fp)


def read_json(fp):
    """
    Reads the fp json file.
    """
    with open(fp, "r") as fp:
        return json.load(fp)


def nested_dict_to_tuple(d, keys=None, key_only=False):
    """Converts a nested dictionnary of any depth to a list of tuples.

    Args:
        d (dictionnary): nested dictionnary of the form
            {1: {1a: A, 1b: B}, 2:{2a: {2a1: C}}}.
        keys (list, optional): key placeholder. Used for recursion,
            do not use it when calling the function. Defaults to None.
        key_only (boolean, optional): whether to keep only keys or add value.
            Defaults to False

    Returns:
        list: list of tuples in the form:
        [(1, 1a, A), (1, 1b, B), (2, 2a, 2a1, C)]
    """
    result = []
    if keys is None:
        keys = []

    for k, v in d.items():
        if isinstance(v, dict):
            result.extend(
                nested_dict_to_tuple(v, keys + [k], key_only=key_only))
        else:
            if key_only:
                result.append(tuple(keys + [k]))
            else:
                result.append(tuple(keys + [k, v]))
    return result


def write_to_nested_dict(d, value, *args):
    """Writes value to a nested dictionnary with key specified in args.

    Performs operation in place (modifies argument dict directly).
    Usage: write_to_nested_dict(dict, my_value, *("key_1", "key_2"))

    Args:
        d (dictionnary): nested dictionnary of the form
            {1: {1a: A, 1b: B}, 2:{2a: {2a1: C}}}.
        value (any): value to add.
    """
    if args:
        el = d[args[0]]
        if len(args) == 1:
            d[args[0]] = value
            return
        else:
            write_to_nested_dict(el, value, *args[1:])


def gridsearch_params_product(*args):
    """Recursive function taking a list of parameters tuples as input,
    returning their cross product.

    Usage: gridsearch_params_product(*nested_dict_to_tuple(params_dict))

    Input: [("1", "a", [0, 1]), ("2", "b", "c", [2, 3])]
    Output: [
        [("1", "a", 0), ("2", "b", "c", 2)],
        [("1", "a", 1), ("2", "b", "c", 2)],
        [("1", "a", 0), ("2", "b", "c", 3)],
        [("1", "a", 1), ("2", "b", "c", 3)],
    ]

    Returns:
        iter: cross product of parameters.
    """
    if not args:
        return iter(((), ))
    return (items + ((*args[-1][:-1], item), )
            for items in gridsearch_params_product(*args[:-1])
            for item in args[-1][-1])


def print_dict(dictionnary, logger=None, logging_level="info"):
    """Prints/Logs dictionnary in a nice way.

    For key, capitalizes and split the underscores '_'.
    For values, formats integers and float numbers.

    Args:
        dictionnary (dict): cictionnary to print.
        logger (logger, optional): logger if available. Defaults to None.
        logging_level (str, optional): logger level. Defaults to "info".
    """
    def format_key(key):
        return (" ".join(key.split("_"))).capitalize()

    def format_value(value):
        if isinstance(value, int):
            if value < 9999:
                return "{:05d}"
            else:
                return "{:.3e}"
        elif isinstance(value, float):
            if value < 1 and math.floor(value * 10**3) != 0:
                return "{:.3f}"
            else:
                return "{:.3e}"
        else:
            return "{}"

    print_str = ", ".join([("{}: " + format_value(v)).format(format_key(k), v)
                           for k, v in dictionnary.items()])
    if logger is not None:
        log_fn_dict = {
            "info": logger.info,
            "debug": logger.debug,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical
        }
        log_fn = log_fn_dict.get(logging_level, logger.info)
        log_fn(print_str)
    else:
        print(print_str)


def merge_metrics(previous_metrics: dict, metrics_to_add: dict, tag=""):
    """Merges experiment stored metrics with newly computed metrics.
    Will append to previously existing metrics if possible,
    else create new keys.

    Usage:
    previous_metrics: None
    metrics_to_add: {"epoch": 1, "accuracy": 0.5}
    -> {"epoch": [1], "accuracy": [0.5]}

    previous_metrics: {"epoch": [1], "accuracy": [0.5]}
    metrics_to_add: {"epoch": 2, "accuracy": 0.6}
    -> {"epoch": [1, 2], "accuracy": [0.5, 0.6]}

    Args:
        previous_metrics (dict): dictionnary of previously defined metrics.
            None if first iteration.
        metrics_to_add (dict): dictionnary of metrics to add.
            Keys must be strings.
        tag (str, optional): additional tag to add to the metric keys.
            Exemple: add '_test' tag for test evaluation. Defaults to "".

    Returns:
        dict: merged dictionnaries.
    """
    if previous_metrics is None:
        return {k + tag: [v] for k, v in metrics_to_add.items()}
    else:
        for k, v in metrics_to_add.items():
            if k + tag in previous_metrics:
                previous_metrics[k + tag].append(v)
            else:
                previous_metrics[k + tag] = [v]
        return previous_metrics


class Manager():
    def __init__(self, args, config):
        self.output_path = None
        self.root = None

    def get_project_root(self):
        return os.path.dirname(Path(__file__).parent)

    def initialize_logging(self, args):
        self.logging_level = self.get_logging_level(args)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            level=self.logging_level,
                            handlers=[
                                logging.FileHandler(
                                    os.path.join(self.output_path, "logging",
                                                 "exp.log")),
                                logging.StreamHandler()
                            ])

    def get_logging_level(self, args):
        logging_level_dict = {1: logging.INFO, 2: logging.DEBUG}
        logging_level = logging_level_dict.get(args.verbose, logging.WARNING)
        return logging_level

    @staticmethod
    def copytree(source, dest, filter_str=None, symlinks=False, ignore=None):
        for item in os.listdir(source):
            if not any([re.match(pattern, item) for pattern in filter_str]):
                s = os.path.join(source, item)
                d = os.path.join(dest, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)


class ExperimentManager(Manager):
    def __init__(self, args, config):
        self.config = self.update_config_from_args(args, config)
        if config["experiment"]["ghost"]:
            return
        self.output_path = self.generate_output_path_name(args, config)
        self.root = self.get_project_root()
        self.generate_output_folders(config)

        self.save_weigth = True if (config["experiment"].get("monitor_training") and 
                                    config["experiment"]["monitor_training"].get("save_weight", False)) else False
        self.save_grad = True if (config["experiment"].get("monitor_training") and 
                                    config["experiment"]["monitor_training"].get("save_grad", False)) else False

        self.initialize_logging(args)

        self.archive_source_code()
        self.save_config(config)

    @staticmethod
    def generate_output_path_name(args, config):
        output_folder = config["experiment"]["output_folder"]
        config_folder = args.config + args.tag
        dataset_folder = config["dataset"]["name"]
        if config["dataset"]["params"].get("preprocessing"):
            dataset_folder += "_" + \
                config["dataset"]["params"].get("preprocessing")
        timestamp = datetime.now().strftime(format="%m%d%y_%H%M%S")
        return os.path.join(output_folder, config_folder, dataset_folder,
                            timestamp)

    @staticmethod
    def update_config_from_args(args, config):
        if "repetitions" not in config["experiment"]:
            config["experiment"]["repetitions"] = 1
        if args.dataset is not None: config["dataset"]["name"] = args.dataset
        if args.rep is not None: config["experiment"]["repetitions"] = args.rep
        if args.sampling_ratio is not None: config["preprocessor"]["params"]["ratio"] = args.sampling_ratio
        if args.preprocessor is not None: config["preprocessor"]["name"] = args.preprocessor
        if args.features_sampling_ratio is not None: config["preprocessor"]["params"]["ratio_features"] = args.features_sampling_ratio
        if 'task' in args:
            if args.task is not None: config["experiment"]["task"]["name"] = args.task

        return config

    def generate_output_folders(self, config):
        os.makedirs(self.output_path)
        os.makedirs(os.path.join(self.output_path, "source_code"))
        os.makedirs(os.path.join(self.output_path, "logging"))
        os.makedirs(os.path.join(self.output_path, "results"))
        os.makedirs(os.path.join(self.output_path, "saved_model"))
        if config["experiment"]["repetitions"] > 1:
            for rep in range(1, config["experiment"]["repetitions"] + 1):
                os.makedirs(
                    os.path.join(self.output_path, "saved_model",
                                "rep_{}".format(rep)))
                os.makedirs(
                    os.path.join(self.output_path, "results",
                                "rep_{}".format(rep)))
        if config["experiment"].get("monitor_training") and config["experiment"][
                "monitor_training"].get("save_weight", False):
            os.makedirs(
                os.path.join(self.output_path, "monitor_training", "weights"))
        if config["experiment"].get("monitor_training") and config["experiment"][
                "monitor_training"].get("save_grad", False):
            os.makedirs(
                os.path.join(self.output_path, "monitor_training", "grads"))
    

    def archive_source_code(self):
        filter_str = [
            "setup.py", ".gitignore", ".vscode", "data", "outputs",
            "README.md", "requirement.txt", "tests", "tmp", "__pycache__",
            "TODO", "(stats_)", "stats", "(outputs_)", "preprocessing_utils",
            "preprocessing.py", "raw_data", "temp", "configs"
        ]
        save_path = os.path.join(self.root, self.output_path, "source_code")
        self.copytree(self.root, save_path, filter_str)
   
    def save_config(self, config):
        with open(os.path.join(self.output_path, "config.json"), "w") as fp:
            json.dump(config, fp)

    def set_logger(self, logger):
        self.logger = logger

    def monitor_epoch(self, model, labels, epoch):
        if "monitor_training" not in self.config["experiment"]:
            return

        _save_interval = self.config["experiment"]["monitor_training"]["save_interval"]
        save_it_weight = True if self.save_weigth and epoch % _save_interval == 0 else False
        save_it_grad = True if self.save_grad and epoch % _save_interval == 0 else False
        self.save_iteration(model, epoch, save_it_weight, save_it_grad)

    def save_iteration(self, model, epoch, save_it_weight, save_it_grad):
        if save_it_weight or save_it_grad:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if save_it_weight:
                        th.save(
                            param,
                            os.path.join(self.output_path, "monitor_training",
                                         "weights",
                                         name + "_{}.pt".format(epoch)))
                    if save_it_grad:
                        th.save(
                            param.grad,
                            os.path.join(self.output_path, "monitor_training",
                                         "grads",
                                         name + "_{}.pt".format(epoch)))
