#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import logging
import os
from utils.main_utils import AVAILABLE_MODEL, load_data
from utils.experiments_utils import (ExperimentManager,
                                     gridsearch_params_product, load_config,
                                     nested_dict_to_tuple, write_to_nested_dict)
from main_unsup import main as main_unsup
from main import main as main_sup
from main_lp import main as main_lp
import argparse
import time

MAIN_FN = {
    "sup": main_sup,
    "unsup": main_unsup,
    "lp": main_lp
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_process", default=None, type=str, help="Name of the main file to use", choices = ["sup", "unsup", "lp"])
    parser.add_argument("--verbose", type=int, default=2, help="Logging level and other information. Higher returns more specific information")
    parser.add_argument("--config", type=str, default="default", help="Name of the configuration file to run experiment from, without .json extention")
    parser.add_argument("--gs_config", type=str, default="default", help="Name of the gs configuration file to run experiment from, without .json extention")
    parser.add_argument("--tag", type=str, default="", help="Specify tag to recognize experiment easily")
    parser.add_argument("--gpu", type=int, default=0, help="cpu: <0, gpu: >0 ")
    parser.add_argument("--dataset", default=None, type=str, help="dataset name")
    parser.add_argument("--rep", default=None, type=int, help="Number of repetitions of experiment")
    parser.add_argument("--task", default=None, type=str, help="Downstream task to use embeddings for. Either 'link_prediction' or 'node_classification'", choices=["link_prediction", "node_classification"])
    parser.add_argument("--preprocessor", default=None, type=str, help="Specify preprocessor. Either 'sampler' or 'features_sampler'", choices=["sampler", "features_sampler"])
    parser.add_argument("--sampling_ratio", default=None, type=float, help="Sampling ratio for sampling preprocessing")
    parser.add_argument("--features_sampling_ratio", default=None, type=float, help="Sampling ratio for sampling nodes features preprocessing")

    args = parser.parse_args()

    params_dict = load_config(os.path.join("configs", "gs", args.gs_config + ".json"))

    params_list = nested_dict_to_tuple(params_dict)
    params_product = gridsearch_params_product(*params_list)

    main_fn = MAIN_FN[args.training_process]

    config = load_config(os.path.join("configs", args.config + ".json"))
    config = update_config_from_args(args, config)

    data_dict = load_data(config, AVAILABLE_MODEL[config["model"]["name"]], logger=None)

    for params in list(params_product):
        config = load_config(os.path.join("configs", args.config + ".json"))

        for *param_keys, param_value in params:
            write_to_nested_dict(config, param_value, *param_keys)

        exp_manager = ExperimentManager(args, config)

        logger = logging.getLogger(__name__)
        exp_manager.set_logger(logger)

        try:
            main_fn(data_dict, args, config, exp_manager, logger)
        except Exception as e:
            logger.exception("Exception occurred during main task : {}".format(e))

        time.sleep(1)
