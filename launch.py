#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

from subprocess import run
import argparse

config = "gcn"
script = "main.py"

datasets = [
    ("pubmed", "small", "labelled"),
    ("cs", "medium", "labelled"),
    ("amazon_co_buy_computer", "medium", "labelled"),
    ("reddit", "large", "labelled")
]

processor_type = {
    "main.py": ["sampler", "features_sampler"],
    "main_lp.py": ["sampler", "features_sampler"],
    "main_ae.py": ["sampler", "features_sampler"],
    "main_unsup.py": ["sampler"]
}

available_tasks = {
    "labelled": ["node_classification", "link_prediction"],
    "unlabelled": ["link_prediction"]
}

ratio_list = {
    "small": [0.5, 0.1, 0.01],
    "medium": [0.5, 0.1, 0.01, 0.005],
    "large": [0.5, 0.1, 0.01, 0.005, 0.001]
}

features_ratio_list = {
    "sampler": [0],
    "features_sampler": [0.75, 0.5, 0.1]
}


def format_tag(task, preprocessor, ratio, features_ratio):
    preprocessor_mapping = {"sampler": "s", "features_sampler": "fs"}
    task_mapping = {"link_prediction": "lp", "node_classification": "nc"}
    if task is not None and preprocessor is not None:
        tag = "_{}_{}_{}".format(task_mapping[task], preprocessor_mapping[preprocessor], str(ratio))
    elif task is not None and preprocessor is None:
        tag = "_{}".format(task_mapping[task])
    elif task is None and preprocessor is not None:
        tag = "_{}_{}".format(preprocessor_mapping[preprocessor], str(ratio))
    else:
        tag = ""
    if preprocessor == 'features_sampler':
        tag = tag + "_{}".format(str(features_ratio))
    return tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--script", type=str, default=None, help="Main script to run experiment from, with .py extension", choices = ["main.py", "main_ae.py", "main_lp.py", "main_unsup.py"])
    parser.add_argument("--config", type=str, default=None, help="Name of the configuration file to run experiment from, without .json extention")
    parser.add_argument("--gpu", type=int, default=None, help="cpu: <0, gpu: >0 ")
    parser.add_argument("--datasets_mask", default=[1, 1, 1, 1],  nargs='+', type=int, help="Specify which datasets to use. 1 at position i if you want to run the script on element i of the datasets list.")
    parser.add_argument("--rep", default=None, type=int, help="Number of repetitions of experiment")
    parser.add_argument("--task", default=None, type=str, help="Downstream task to use embeddings for. Either 'link_prediction' or 'node_classification'", choices=["link_prediction", "node_classification", "classic_tabular_model_nc"])
    parser.add_argument("--preprocessor", default=None, type=str, help="Specify preprocessor. Either 'sampler', or 'features_sampler'", choices=["sampler", "features_sampler"])

    args = parser.parse_args()

    if args.script is not None: script = args.script
    if args.config is not None: config = args.config

    assert len(args.datasets_mask) == len(datasets), 'datasets_mask argument must have the same lenght as datasets'
    datasets = [datasets[i] for i in range(len(datasets)) if args.datasets_mask[i]==1]

    for (dataset, dataset_size, dataset_type) in datasets:
        processors = processor_type[script]
        if args.preprocessor is not None:
            processors = [p for p in processors if p == args.preprocessor]
            if len(processors) == 0:
                print('Run baseline only because wanted processors is not available for the main function.')

        if script in ["main.py", "main_lp.py"]: tasks = [None]
        # If main.py is used, NC only possible solution
        # If main_lp.py is used, LP only possible solution
        else:
            tasks = available_tasks[dataset_type]
            if args.task is not None:
                tasks = [t for t in tasks if t == args.task]
                assert len(tasks) != 0, 'give a relevant task for the dataset_type'
        ratios = ratio_list[dataset_size]

        # Run baseline
        for task in tasks:
            cmd = ["python", script,
                   "--config",  config,
                   "--dataset", dataset,
                   "--tag", format_tag(task, None, None, None)]

            if args.gpu is not None: cmd += ["--gpu", str(args.gpu)]
            if args.rep is not None: cmd += ["--rep", str(args.rep)]
            if task is not None: cmd += ["--task", task] 

            print("\n\n*#*#**#*#* Running baseline {} with cfg: {} on {}, task: {}*#*#**#*#*\n\n".format(script, config, dataset, task))
            try:
                run(cmd)
            except Exception as e:
                print("Exception occured, continue.. {}".format(e))

        # Run experiment
        for preprocessor in processors:
            features_ratios = features_ratio_list[preprocessor]
            for task in tasks:
                for ratio in ratios:
                    for features_ratio in features_ratios:
                        cmd = ["python", script,
                               "--config",  config,
                               "--dataset", dataset,
                               "--tag", format_tag(task, preprocessor, ratio, features_ratio),
                               "--preprocessor", preprocessor,
                               "--sampling_ratio", str(ratio),
                               "--features_sampling_ratio", str(features_ratio)]

                        if args.gpu is not None: cmd += ["--gpu", str(args.gpu)]
                        if args.rep is not None: cmd += ["--rep", str(args.rep)]
                        if task is not None: cmd += ["--task", task]

                        print("\n\n*#*#**#*#* Running {} with cfg: {} on {} (prepro: {} / r: {}), task: {}*#*#**#*#*\n\n".format(script, config, dataset, preprocessor, ratio, task))
                        try:
                            run(cmd)
                        except Exception as e:
                            print("Exception occured, continue.. {}".format(e))
