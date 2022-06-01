# Experiment pipeline for efficient Graph Representation Learning models robustness evaluation

**Authors:** Marine NEYRET (Institut Louis Bachelier DataLab), Pierre SEVESTRE (Société Générale, DataLab IGAD)

For more details, see the ArXiv's version of the [paper](https://arxiv.org/abs/2205.09648)

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick demo](#quick-demo)
- [How to use](#how-to-use)
  - [Config files](#config-files)
  - [Supervised training for node classification](#supervised-training-for-node-classification)
  - [Supervised training for link prediction](#supervised-training-for-link-prediction)
  - [Unsupervised training for node classification & link prediction](#unsupervised-training-for-node-classification--link-prediction)
  - [Additional scripts](#additional-scripts)
    - [Gridsearch](#gridsearch)
    - [Large-scale experiments](#large-scale-experiments)
    - [Analyse results](#analyse-results)
- [License](#license)

## Introduction

In this work, we propose an experimental pipeline to challenge the robustness of Graph Representation Learning (GRL) models to properties encountered on transactional graphs from a bank perspective. The bank does not have a complete view of all transactions being performed on the market, and the resulting subgraph is sparser than the complete graph of transactions, restricted to transactions involving one of its clients, resulting in the following specificities:
- **Graph sparsity**  A large part of the transactions occuring between non-clients are not available from the bank point of view;
- **Asymmetric node information** More attributes are available for the clients than for non-clients.

To analyze the behavior of GRL models when dealing with transactional graph specificities, two preprocessors are created to deteriorate a full graph to match either graph sparsity or asymmetric node information characteristic.

While results are reported on four commonly used graph in the litterature (Pubmed, CoAuthor CS, Amazon Computer and Reddit), the pipeline can easily be used on other graphs.

<ins>Preprocessors details:</ins>

The *sampler* processor deals with graph sparsity:

1. Sample a portion of nodes that will act as client nodes,
2. Remove any edge not connected to this subset of nodes.

The *features_sampler* handles asymmetric node information:

1. Sample a portion of nodes that will act as client nodes, the remaining ones being non-client nodes,
2. Remove any edge not connected to this subset of nodes,
3. Deteriorate a portion of node features for non-clients nodes.

**Parameters:**
- `sampling_ratio`: portion of nodes to be considered as clients,
- `features_sampling_ratio` (only for *features_sampler*): portion of node features to be deteriorated.


You can create your own preprocessor to experiment any other graph characteristic in `preprocessor.py` by inheriting from the `Preprocessor` class. You must define the internal function `_process` that performs the graph modifications.

## Installation

Implemented and tested on the following environment:
- Linux (RHEL v7.9) with Tesla V100-PCIE-16GB 
- Python 3.7
- Pytorch 1.5.1 and DGL 0.6.1
  
To use this experimental pipeline:
1. Clone the repository
2. Install the dependencies. The required Python libraries are listed in `requirements.txt`: run `pip install -r requirements.txt`

## Quick demo

To run a GCN model (config defined in `config/gcn.json`) for node classification on the full Pubmed dataset:

```console
python main.py --config gcn --dataset pubmed --rep 1
```

Expected output:
```console
INFO | 28-Jan-22 17:00:20 | Computing test metrics..
INFO | 28-Jan-22 17:00:20 | Accuracy: 0.786, Precision macro: 0.778, Recall macro: 0.788, F1 macro: 0.781
```

To run a GraphSage model (config defined in `config/graphsage_lp.json`) for link prediction on the sampled Pubmed dataset (with 0.5 ratio):

```console
python main_lp.py --config graphsage_lp --dataset pubmed --preprocessor sampler --sampling_ratio 0.5 --rep 1 
```

Expected output:
```console
INFO | 28-Jan-22 17:18:00 | Computing test metrics..
INFO | 28-Jan-22 17:18:00 | Precision lp: 0.542, Recall lp: 0.951, Accuracy lp: 0.573, F1 lp: 0.690, Auc lp: 0.741, Ap lp: 0.701, Nb removed edges: 00000, Accuracy removed edges: 00000
```
## How to use 

3 main files are available to run experiments for the different training procedure and downstream tasks, consisting in:
- `load_data`: starting from a DGL dataset, applies the preprocessor if needed, computes the train/validation/test masks, and stores the usefull data for training and evaluation into a dictionnary,
- `main`: trains one model and evaluates its performances. 

Results, logging, checkpoints models, source code and config are automatically saved according to the path registered in the configuration file.

### Config files

The most important argument of main files is *config*: the configuration file to run experiment from, stored in the `config` folder.

Each experiment is associated with a configuration file describing its characteristics. The configuration files should be saved in `json` format under `./configs/<config_name>.json`. The description of such configuration is available below.

<details><summary><b>Config description</b></summary>

```json
{
    "dataset": {
        "name": "<name>" // Name of the dataset to run experiments on, from AVAILABLE_DATASET in utils/main_utils.py
    },
    "preprocessor":{
        "name": "<name>", // Preprocesor name, from preprocessor_dict in utils/main_utils.py or null 
                          // for baseline
        "params": {
            "ratio": float, // Portion of client nodes
            "method": "<mehod_name>", // Features deterioration method, either 'zeros', 'random', 
                            // 'drop' or 'imputation'.
            "ratio_features": float // Portion of features to deteriorate
        }
    },
    "experiment": {
        "ghost": false, 
        "output_folder": "<name>", // Name of the folder in which experiments outputs are stored
        "use_saved_datadict": bool, // True to reuse preprocessed data for identical configuration
        "ckpt_save_interval": 1000, // Checkpoint interval 
        "repetitions":10, // Number of independant model trainings and evaluation
        "val_metrics_interval": 2, // Interval between two validation metrics computations        
        "monitor_training": { // For advance training monitoring
            "save_weight": false,
            "save_grad": false,
            "save_interval": 1
        },
        "task": {
            "name": "<task_name>", // Name of the downstream task, either 'node_classification' 
                                   // or 'link_prediction'
            "params": {// Parameters associated with task. See example configurations provided.
            }
        }   
    },

    "model": { 
        "name": "<model_name>", // Name of the model, from AVAILABLE_MODEL in utils/main_utils.py
        "features": true, // Whether model requires node features or not
        "params": {// Parameters of the model, must be available in model.py class definition
        }
    },
    "training":{
        "batch_size": int,
        "n_epochs": int,
        "loss":{
            "use_weights": bool
        },
        "optimizer":{
            "lr": float,
            "weight_decay": float
        },
        "early_stopping":{
            "patience": int,
            "delta": float
        }
    },
    "evaluation":{// Downstream task for link_prediction
        "model": {
            "name": "log_reg",
            "params":{
                "penalty": "none",
                "solver":"newton-cg"
            }
        }
    }
}

```
</details>

### Supervised training for node classification 

- **Script:** `main.py`
- **Available models and corresponding config:** 
  
  Implemented for the models below, but can be easily extended to any `nn.Module` with appropriate `forward` method

    | Model name | Config name |
    | ---------- | ----------- | 
    | [GCN](https://arxiv.org/abs/2003.01171)        | `gcn`         | 
    | [GAT](https://arxiv.org/abs/1710.10903)        | `gat`         | 
    | [GraphSage](https://arxiv.org/abs/1706.02216)  | `graphsage`   | 


```
$ python main.py --help
usage: main.py [-h] [--verbose VERBOSE] [--config CONFIG] [--tag TAG]
               [--gpu GPU] [--dataset DATASET] [--rep REP]
               [--preprocessor {sampler,features_sampler}]
               [--sampling_ratio SAMPLING_RATIO]
               [--features_sampling_ratio FEATURES_SAMPLING_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --verbose VERBOSE     Logging level and other information. Higher value
                        returns more specific information
  --config CONFIG       Name of the configuration file to run experiment from,
                        without .json extension
  --tag TAG             Specify tag to recognize experiment easily
  --gpu GPU             cpu: <0, gpu: >=0
  --dataset DATASET     Dataset name
  --rep REP             Number of repetitions of the experiment
  --preprocessor {sampler,features_sampler}
                        Specify preprocessor. Either 'sampler', or
                        'features_sampler'
  --sampling_ratio SAMPLING_RATIO
                        Sampling ratio to identify the portion of client nodes
  --features_sampling_ratio FEATURES_SAMPLING_RATIO
                        Sampling ratio for nodes features deterioration
``` 

### Supervised training for link prediction

- **Script:** `main_lp.py`
- **Available models and corresponding config:**

    Implemented for the models below, but can be easily extended to any `nn.Module` with appropriate `forward` method

    | Model name | Config name  |
    | ---------- | ------------ | 
    | [GCN](https://arxiv.org/abs/2003.01171)        | `gcn_lp`       | 
    | [GAT](https://arxiv.org/abs/1710.10903)        | `gat_lp`       | 
    | [GraphSage](https://arxiv.org/abs/1706.02216)  | `graphsage_lp` |


```
python main_lp.py --help
usage: main_lp.py [-h] [--verbose VERBOSE] [--config CONFIG] [--tag TAG]
                  [--gpu GPU] [--dataset DATASET] [--rep REP]
                  [--preprocessor {sampler,features_sampler}]
                  [--sampling_ratio SAMPLING_RATIO]
                  [--features_sampling_ratio FEATURES_SAMPLING_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --verbose VERBOSE     Logging level and other information. Higher value
                        returns more specific information
  --config CONFIG       Name of the configuration file to run experiment from,
                        without .json extension
  --tag TAG             Specify tag to recognize experiment easily
  --gpu GPU             cpu: <0, gpu: >=0
  --dataset DATASET     Dataset name
  --rep REP             Number of repetitions of the experiment
  --preprocessor {sampler,features_sampler}
                        Specify preprocessor. Either 'sampler', or
                        'features_sampler'
  --sampling_ratio SAMPLING_RATIO
                        Sampling ratio to identify the portion of client nodes
  --features_sampling_ratio FEATURES_SAMPLING_RATIO
                        Sampling ratio for nodes features deterioration
```

### Unsupervised training for node classification & link prediction

- **Script:** `main_unsup.py`
- **Available models and corresponding config:**

    | Model name | Config name           | 
    | ---------- | --------------------- | 
    | [DeepWalk](https://arxiv.org/abs/1403.6652)   | `deepwalk` / `deepwalk_lp` | 
    | [Node2Vec](https://arxiv.org/abs/1607.00653)   | `node2vec` / `node2vec_lp` | 

- **Dowstream task:** to be specified using `--task` argument

```
python main_unsup.py --help
usage: main_unsup.py [-h] [--verbose VERBOSE] [--config CONFIG] [--tag TAG]
                     [--gpu GPU] [--dataset DATASET] [--rep REP]
                     [--task {link_prediction,node_classification}]
                     [--preprocessor {sampler,features_sampler}]
                     [--sampling_ratio SAMPLING_RATIO]
                     [--features_sampling_ratio FEATURES_SAMPLING_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --verbose VERBOSE     Logging level and other information. Higher value
                        returns more specific information
  --config CONFIG       Name of the configuration file to run experiment from,
                        without .json extension
  --tag TAG             Specify tag to recognize experiment easily
  --gpu GPU             cpu: <0, gpu: >=0
  --dataset DATASET     Dataset name
  --rep REP             Number of repetitions of the experiment
  --task {link_prediction,node_classification}
                        Downstream task to use embeddings for. Either
                        'link_prediction' or 'node_classification'
  --preprocessor {sampler,features_sampler}
                        Specify preprocessor. Either 'sampler', or
                        'features_sampler'
  --sampling_ratio SAMPLING_RATIO
                        Sampling ratio to identify the portion of client nodes
  --features_sampling_ratio FEATURES_SAMPLING_RATIO
                        Sampling ratio for nodes features deterioration
```

### Additional scripts

#### Gridsearch

You could use `gridsearch.py` script to perform hyperparameter search on a dataset. Example below for *GCN* on Pubmed:

```console
python gridsearch.py --training_process sup --config gcn --gs_config gs_gcn --dataset pubmed
```

A second configuration file is needed to gridsearch parameters, providing the lists of parameters to search on. The description of such configuration is available below.

<details><summary><b>Gridsearch config example</b></summary>

```json
{
    "model": {
        "params": { // Parameters of the model, must be available in model.py class definition
            "<name_parameter>": [value, value, value],
            ...
        }
    },
    "training": {// Parameters for training
        "<param_categ>":{
            "<name_parameter>": [value, value, value],
            ...
        }
    }
}

```
</details>

#### Large-scale experiments

You could use `launch.py` to run one model on several datasets and for several prepocessors. To run all the experiments for a *GCN* model:

```console
python launch.py --script main.py --config gcn
```

#### Analyse results

You can find a Jupyter notebook, in the notebook folder, to:

- merge all the results within an excel file,
- plot graphs to see performance evolution with respect to *sampler* or *features_sampler*.

## Acknowledgement
This research was conducted within the "Data Science for Banking Audit" research program, under the aegis of the Europlace Institute of Finance, a joint program with the General Inspection at Société Générale.

  
## License
Under the terms of the BSD 2-Clause License.
  
<details><summary> <b>Third Party Libraries </b></summary>
  
| Component   | Version     | License  |
| ----------- | ----------- |----------|
| [nsim](https://pypi.org/project/gensim/3.8.3/#history) | 3.8.3 |[GNU LGPL 2.1](https://pypi.org/project/gensim/3.8.3/#history) |
| [GPUtil](https://github.com/anderskm/gputil/tree/v1.4.0) | 1.4.0 | [MIT](https://github.com/anderskm/gputil/blob/master/LICENSE.txt) |
| [karateclub](https://pypi.org/project/karateclub/#history)| 1.0.21 | [MIT](https://pypi.org/project/karateclub/#history) |
| [matplotlib](https://pypi.org/project/matplotlib/#history) | 3.3.2 | [Python Software Foundation](https://pypi.org/project/matplotlib/#history) |
| [networkx](https://github.com/networkx/networkx/tree/networkx-2.5) | 2.5 | [BSD](https://github.com/networkx/networkx/blob/main/LICENSE.txt) |
| [node2vec](https://github.com/eliorc/node2vec/tree/v0.4.0) | 0.4.0 | [MIT](https://github.com/eliorc/node2vec/blob/master/LICENSE) |
| [nodeventors](https://github.com/VHRanger/nodevectors/tree/0.1.23) | 0.1.23 | [MIT](https://github.com/VHRanger/nodevectors/blob/master/LICENSE.txt) |
| [numpy](https://github.com/numpy/numpy/tree/v1.19.0) | 1.19.0 | [BSD-3](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [pandas](https://github.com/pandas-dev/pandas/tree/v1.1.5) | 1.1.5 | [BSD-3](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| [pyarrow](https://github.com/apache/arrow/tree/apache-arrow-0.13.0) | 0.13.0 | [Apache v2](https://github.com/apache/arrow/blob/master/LICENSE.txt) |
| [pyrsistent](https://github.com/tobgu/pyrsistent/tree/v0.16.0) | 0.16.0 | [MIT](https://github.com/tobgu/pyrsistent/blob/master/LICENSE.mit) |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn/tree/0.23.2) | 0.23.2 | [BSD-3](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [torch](https://pypi.org/project/torch/#history) | 1.5.1+cu101 | [BSD-3](https://pypi.org/project/torch/#history) |
| [torchvision](https://pypi.org/project/torchvision/#history) | 0.6.1+cu101 | [BSD](https://pypi.org/project/torchvision/#history) |
| [tqdm](https://github.com/tqdm/tqdm/tree/v4.50.2) | 4.50.2 | [MIT License, Mozilla Public License 2.0 (MPL 2.0) (MPLv2.0, MIT Licences)](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| [dgl-cu101](https://pypi.org/project/dgl-cu101/#history) | 0.6.1 | [Apache Software License (APACHE)](https://pypi.org/project/dgl-cu101/#history) |
</details>



