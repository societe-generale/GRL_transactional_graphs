#!Copyright (c) 2022, Société Générale.
#!All rights reserved.

#!This source code is licensed under the BSD 2-clauses license found in the
#!LICENSE file in the root directory of this source tree.  

import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import itertools as it
from utils.experiments_utils import print_dict, read_json

METRIC_NAMES_LABELS = {
    'accuracy_test': 'Accuracy - Test',
    'accuracy_lp_test': 'Accuracy - Test',
    'f1_macro_test': 'Macro F1 - Test',
    'auc_lp_test': 'AUC - Test',
    'auc_test': 'AUC - Test'
}

#############################################################################
# Pure utils functions
#############################################################################


def pad_to_max_len(lists, value=np.nan):
    """Pad lists to the length of the length max length

    Args:
        lists (iterrable): Iterrable of lists to pad
        value (any, optional): Value to pad with. Defaults to None.
    """

    max_len = max(len(x) for x in lists)
    out_list = []
    for l in lists:
        out_list.append(l + [value]*(max_len-len(l)))
    return out_list

##############################################################################
# Functions about metrics files
##############################################################################


def decode_tag(tag, verbose=0):
    """
    Complex regex function to decode tag experience name and retreive experience settings.
    Outputs model, and if available, evaluation task (lp or nc), sampler and sampler ratio. 
    """

    pattern_sampler = "\w+_(s|fs)_(0.\d+)$"
    pattern_sampler_task = "^(\w+)_(nc|lp)"
    pattern_sampler_baseline = "^(([^(_lp)\s]+)_(nc|lp))"
    pattern_baseline = "^(([^(_lp)\s]+)_(nc|lp))$"
    pattern_task = "^(\w+)_(nc|lp)$"
    pattern_sampler_baseline_notask = "^(\w+)_(s|fs)_(0.\d+)"
    pattern_baseline_nosampler_notask = "^(\w+)$"

    out = {"tag": None, "model": None, "task": None, "sampler": None, "ratio": None}
    match_sampler = re.match(pattern_sampler, tag)
    if match_sampler:
        match_sampler_baseline = re.match(pattern_sampler_baseline, tag)
        match_sampler_task = re.match(pattern_sampler_task, tag)
        if match_sampler_task and match_sampler_baseline:
            out.update({"tag": tag, "model": match_sampler_task.group(1), "task": match_sampler_task.group(2), "sampler": match_sampler.group(1), "ratio": float(match_sampler.group(2))})
            if verbose: print_dict(out)
        else:
            match_sampler_baseline_notask = re.match(pattern_sampler_baseline_notask, tag)
            out.update({"tag": tag, "model": match_sampler_baseline_notask.group(1), "sampler": match_sampler.group(1), "ratio": float(match_sampler.group(2))})
            if verbose: print_dict(out)
    else:
        match_baseline = re.match(pattern_baseline, tag)
        match_task = re.match(pattern_task , tag)
        if match_task and not match_baseline:
            out.update({"tag": tag, "model": match_task.group(1), "task": match_task.group(2)})
            if verbose: print_dict(out)
        else:
            match_baseline_nosampler_notask = re.match(pattern_baseline_nosampler_notask, tag)
            out.update({"tag": tag, "model": match_baseline_nosampler_notask.group(1)})
            if verbose: print_dict(out)
    return out


def encode_tag(model, task, preprocessor, ratio, ratio_nf):
    """Create tag from experiment params.

    Args:
        model (str):  model name.
        task (str): task full name (link_prediction or node_classfication).
        preprocessor (str): preprocessor name (None, 'sampler', 'features_sampler').
        ratio (float): percentage of nodes considered as clients.
        ratio_nf (float): percentage of nodes modified features.

    Returns:
        str: tag corresponding to experiment params.
    """
    preprocessor_mapping = {"sampler": "s", "features_sampler": "fs"}
    task_mapping = {"link_prediction": "lp", "node_classification": "nc"}
    if task is not None:
        if preprocessor == "sampler":
            return "{}_{}_{}_{}".format(model, task_mapping[task], preprocessor_mapping[preprocessor], str(ratio))
        elif preprocessor is None:
            return "{}_{}".format(model, task_mapping[task])
        else:
            return "{}_{}_{}_{}_{}".format(model, task_mapping[task], preprocessor_mapping[preprocessor], str(ratio), str(ratio_nf))
    elif task is None:
        if preprocessor == "sampler":
            return "{}_{}_{}".format(model, preprocessor_mapping[preprocessor], str(ratio))
        elif preprocessor is None:
            return "{}".format(model)
        else:
            return "{}_{}_{}_{}".format(model, preprocessor_mapping[preprocessor], str(ratio), str(ratio_nf))


def merge_repetitions_metrics(metrics, reduce):
    """
    Aggregate metrics if experimentwith multiple repetition.
    The type of aggregation is given by reduce 
    parameter (str) 'mean' or 'std' are possible.
    """
    metrics = pd.DataFrame(metrics).applymap(lambda x : x[0])
    metrics = metrics.loc[:, metrics.dtypes != object].copy()
    if reduce == 'mean':
        return metrics.apply(lambda x : np.nanmean(x)).to_dict()
    elif reduce == 'std':
        return metrics.apply(lambda x : np.nanstd(x)).to_dict()
    else:
        raise NotImplementedError


def get_exp_metrics(model, task, preprocessor, ratio, ratio_nf, dataset, output_path, order="latest", reduce="mean"):
    """
    Find and load json metrics file according to experiment params.

    Args:
        model (str): model name.
        task (str): task full name (None, link_prediction or node_classfication).
        preprocessor (str): preprocessor name (None, 'sampler', 'features_sampler').
        ratio (float): percentage of nodes considered as clients.
        ratio_nf (float): percentage of modified nodes features.
        dataset (str): dataset name using our convention. See main_utils.
        output_path (str): path where experiment results are saved.
        order (str, optional): method to use when one experiment is available several times, "latest" is the
        only implemented method and reads the latest experiment. Defaults to "latest".
        reduce (str, optional): when multiple repetitions have been performed, aggregate them with specified 
        method. Defaults to "mean". Other possibility 'std'.

    Raises:
        NotImplementedError: when order is not equal to "latest" or method not in ['mean', 'std']

    Returns:
        dict: metrics dictionnary
    """
    tag = encode_tag(model, task, preprocessor, ratio, ratio_nf)
    exp_dir = os.path.join(output_path, tag, dataset)
    exp_list = os.listdir(exp_dir) if os.path.isdir(exp_dir) else []
    if len(exp_list) == 0:
        print("No experiment found for {} on {}".format(tag, dataset))
        return None

    exp_folder = exp_list[0]
    if len(exp_list) > 1:
        print("Multiple experiments in folder {}, selecting {}".format(tag, order))
        if order == "latest":
            exp_folder = exp_list[-1]
        else:
            raise NotImplementedError
    metric_file = "all_metrics.json"
    if not os.path.isfile(os.path.join(exp_dir, exp_folder, metric_file)):
        metric_file = "metrics.json"
        if not os.path.isfile(os.path.join(exp_dir, exp_folder, metric_file)):
            print("No metric file found in folder {}".format(os.path.join(exp_dir, exp_folder)))
            return None

    metrics = read_json(os.path.join(exp_dir, exp_folder, metric_file))

    # Handle metrics with multiple repetition
    if type(metrics) == list:
        if len(metrics) > 1:
            # Multiple repetitions found, reducing
            metrics = merge_repetitions_metrics(metrics, reduce)
        else:
            metrics = metrics[0]

    return metrics

##############################################################################
# Functions to nicely plot results
##############################################################################


def create_colormap(cname):
    """
    Create list of 6 colors. 
    Name format:
    - custom name for our own colors
    - mpl colors, key of plt.colormaps()
    - seaborn: use "sns_<color_palette_name>". Must use this format since some palettes are contained in mpl
    """
    if cname == "custom1":
        return ['dimgrey', 'indianred', '#463E43', 'tomato', 'brown', 'silver','#ff9999']
    elif cname == "custom2_6":
        return ['#4D385E', '#A10D0B', '#E71915', 'tomato', 'silver', 'dimgrey', '#243851', '#334674'][1:-1]
    elif cname == "custom_2_8":
        return ['#4D385E', '#A10D0B', '#E71915', 'tomato', 'silver', 'dimgrey', '#243851', '#334674']
    elif cname in plt.colormaps():
        return cm.get_cmap(cname)(np.linspace(0, 1, 6))[:,:3]
    elif cname.startswith("sns"):
        return sns.color_palette(cname.split('_')[1])
    else:
        raise NotImplementedError


def ratio_metric_subplots(metric, ratios, x_label, baseline=True, figsize=(12,18), x_freq=10, set_labels=True):
    """Fig, ax creation for single metric plot along ratios.

    Args:
        metric (str): metric name.
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        x_label (str): label for x axis.
        baseline (bool, optional): whether to include baseline. 
        Defaults to True.
        figsize (tuple, optional): size of figure. Defaults to (12,18).
        x_freq (int, optional): frequency of x ticks. Defaults to 10.
        set_labels (bool, optional): whether to print axis labels. 
        Defaults to True.

    Returns:
        tuple: fig, ax objects.
    """
    if baseline: ratios = ['baseline'] + ratios
    fig, ax = plt.subplots()
    if set_labels:
        ax.set_xlabel(x_label)
        ax.set_ylabel("{}".format(METRIC_NAMES_LABELS[metric]))
    ax.yaxis.grid(True)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(ratios)
    return fig, ax


def plot_exp(ax, ratios, ratio_metrics, colors, lst='--', label=None, marker="x", markersize=10, lw=2, alpha=1, color_index=0):
    """Plot evolution of metric as ratio of client decreases for one preprocessor.

    Args:
        ax (AxesSubplot).
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        ratio_metrics (list of float): list of metric values corresponding to the ratio at the same position in ratios.
        colors (list of str): list of colors.
        lst (str, optional): line style. Defauts to '--'.
        label (list of str, optional): list of labels. Defaults to None.
        marker (str, optional): marker to use on graph. Defaults to "+".
        markersize (int, optional): marker size to use on graph. Defaults to 10.
        lw (int, optional): linewidth. Defaults to 2.
        alpha (float, optional): Defaults to 1.
        color_index (int, optional): position of the color to use. Defaults to 0.
    """
    color = colors[color_index+1] if color_index +1 < len(colors) else None
    if len(ax.xaxis.get_major_ticks()) == len(ratios)+1: # Check if baseline displayed
        xrange = range(1, len(ratios)+1)
    else: xrange = range(len(ratios))
    ax.plot(xrange, ratio_metrics, label=label, marker=marker, linestyle=lst, markersize=markersize, lw=lw, alpha=alpha, color=color)


def plot_metric_wrt_models_ratios(models, task, preprocessor, ratios, ratio_nf, dataset, metric, colors, markers, output_path, savefig_path, baseline=True, metric_std=True, verbose=0, title=True, set_labels=True, save=False, dpi=200, lst='--'):
    """Plot evolution of one metric as ratio of client decreases for one preprocessors and several models.

    Args:
        models (list of str): list of model names.
        task (str): task full name (None, link_prediction or node_classfication).
        preprocessor (str): preprocessor name (None, 'sampler', 'features_sampler').
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        ratio_nf (float): percentage of modified nodes features.
        dataset (str): dataset name using our convention. See main_utils.
        metric (str): metric name to plot.
        colors (list of str): list of colors.
        markers (list of str): list of markers.
        output_path (str): path where experiment results are saved.
        savefig_path (str): path where figures can be saved.
        baseline (bool, optional): whether to add baseline corresponding to ratio=0.
        Defaults to True.
        metric_std (bool, optional): whether to add std confidence interval
        on plots. Defaults to True.
        verbose (int, optional): Defaults to 0.
        title (bool, optional): whether to add preprocessor | dataset as title.
        Defaults to False.
        set_labels (bool, optional): whether to print axis labels. 
        Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        dpi (float, optional): the resolution in dots per inch. Defaults to 200.
        lst (str, optional): line style. Defauts to '--'

    Returns:
        tuple: fig, tuple of ax objects.
    """
    fig, ax = ratio_metric_subplots(metric, ratios, "ratio_sampling", baseline=baseline, set_labels=set_labels)
    for index, model in enumerate(models):
        model_ratios_metric = []
        model_ratios_std = []
        for ratio in ratios:
            try:
                if model in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                    metrics = get_exp_metrics(model, None, preprocessor, ratio, ratio_nf, dataset, output_path)
                    metrics_std = get_exp_metrics(model, None, preprocessor, ratio, ratio_nf, dataset, output_path, reduce='std')
                else:
                    metrics = get_exp_metrics(model, task, preprocessor, ratio, ratio_nf, dataset, output_path)
                    metrics_std = get_exp_metrics(model, task, preprocessor, ratio, ratio_nf, dataset, output_path, reduce='std')
                model_ratios_metric.append(metrics[metric])
                model_ratios_std.append(metrics_std[metric])
            except FileNotFoundError:
                if verbose: print("No experiment found: {}".format(encode_tag(model, task, preprocessor, ratio, ratio_nf)))
            except KeyError:
                if verbose: print("Metric: {} not available for {}".format(metric, encode_tag(model, task, preprocessor, ratio, ratio_nf)))

        if baseline:
            if model in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                metrics = get_exp_metrics(model, None, None, None, None, dataset, output_path)
                metrics_std = get_exp_metrics(model, None, None, None, None, dataset, output_path, reduce='std')
            else:
                metrics = get_exp_metrics(model, task, None, None, None, dataset, output_path)
                metrics_std = get_exp_metrics(model, task, None, None, None, dataset, output_path, reduce='std')
            model_ratios_metric = [metrics[metric]] + model_ratios_metric 
            model_ratios_std = [metrics_std[metric]] + model_ratios_std           
            _ratios = [1] + ratios
        else: _ratios = ratios
        plot_exp(ax, _ratios, model_ratios_metric, colors, lst=lst, label=model, color_index=index-1, marker=markers[index], lw=0.75)
        if metric_std:
            ax.fill_between(range(len(_ratios)), np.array(model_ratios_metric) - np.array(model_ratios_std), np.array(model_ratios_metric) + np.array(model_ratios_std), facecolor=colors[index], alpha=.3)
    ax.legend()
    if title: ax.set_title("{} | {}".format(preprocessor, dataset))
    if save: fig.savefig(os.path.join(savefig_path, "{}_{}_{}_{}.png".format(preprocessor, task, dataset, metric)), bbox_inches="tight", dpi=dpi)
    return fig, ax

def plot_metric_wrt_ratio_nf(model, task, preprocessor, ratios, ratios_nf, dataset, metric, colors, markers, output_path, savefig_path, metric_std=False, verbose=0, title=True, save=False, dpi=200, lst='--'):
    """Plot evolution of metric as ratio of modified node features increases, for one model and several node sampling ratios.

    Args:
        model (str): model name.
        task (str): task full name (None, link_prediction or node_classfication).
        preprocessor (str): preprocessor name ('features_sampler').
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        ratios_nf (float): list of nodes features ratios corresponding to
        percentage of modified nodes features.
        dataset (str): dataset name using our convention.
        metric (str): metric to plot.
        colors (list of str): list of 6 colors.
        markers (list of str): list of markers.
        output_path (str): path where experiment results are saved.
        savefig_path (str): path where figures can be saved.
        metric_std (bool, optional): whether to add std confidence interval
        on plots. Defaults to True.
        verbose (int, optional): Defaults to 0.
        title (bool, optional): whether to add model | task | dataset as title.
        Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        dpi (float, optional): the resolution in dots per inch. Defaults to 200.
        lst (str, optional): line style. Defauts to '--'

    Returns:
        tuple: fig, ax objects.
    """
    fig, ax = ratio_metric_subplots(metric, [0]+ratios_nf, "ratio_nf_sampling", baseline=False)
    for index, ratio in enumerate(ratios):
        ratio_nf_metrics = []
        ratio_nf_std = []
        try:
            if model in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                metrics = get_exp_metrics(model, None, "sampler", ratio, None, dataset, output_path)
                metrics_std = get_exp_metrics(model, None, "sampler", ratio, None, dataset, output_path, reduce='std')
            else:
                metrics = get_exp_metrics(model, task, "sampler", ratio, None, dataset, output_path)
                metrics_std = get_exp_metrics(model, task, "sampler", ratio, None, dataset, output_path, reduce='std')
            ratio_nf_metrics.append(metrics[metric])
            ratio_nf_std.append(metrics_std[metric])
        except FileNotFoundError:
            if verbose: print("No experiment found: {}".format(encode_tag(model, task, preprocessor, ratio, ratio_nf)))
        except KeyError:
            if verbose: print("Metric: {} not available for {}".format(metric, encode_tag(model, task, preprocessor, ratio, ratio_nf)))
        
        for ratio_nf in ratios_nf:
            try:
                if model in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                    metrics = get_exp_metrics(model, None, preprocessor, ratio, ratio_nf, dataset, output_path)
                    metrics_std = get_exp_metrics(model, None, preprocessor, ratio, ratio_nf, dataset, output_path, reduce='std')
                else:
                    metrics = get_exp_metrics(model, task, preprocessor, ratio, ratio_nf, dataset, output_path)
                    metrics_std = get_exp_metrics(model, task, preprocessor, ratio, ratio_nf, dataset, output_path, reduce='std')
                ratio_nf_metrics.append(metrics[metric])
                ratio_nf_std.append(metrics_std[metric])
            except FileNotFoundError:
                if verbose: print("No experiment found: {}".format(encode_tag(model, task, preprocessor, ratio, ratio_nf)))
            except KeyError:
                if verbose: print("Metric: {} not available for {}".format(metric, encode_tag(model, task, preprocessor, ratio, ratio_nf)))
        plot_exp(ax, [0]+ratios_nf, ratio_nf_metrics, colors, lst=lst, label=str(ratio), color_index=index, marker=markers[index]) 
        if metric_std:
            ax.fill_between(range(len(ratios_nf)+1), np.array(ratio_nf_metrics) - np.array(ratio_nf_std), np.array(ratio_nf_metrics) + np.array(ratio_nf_std), facecolor=colors[index+1], alpha=.3)
        
    ax.legend(title='ratio_sampling')
    if title: ax.set_title("{} | {} | {}".format(model, task, dataset))
    if save: fig.savefig(os.path.join(savefig_path, "{}_{}_{}_{}.png".format(model, task, dataset, metric)), bbox_inches="tight", dpi=dpi)
    return fig, ax

##############################################################################
# Functions to create merged table of all results
##############################################################################

def summarize_metrics_dataset_task(dataset, MODELS, preprocessors, ratios, ratios_nf, task, metrics_to_save, output_path, baseline, reduce):
    """Concatenate metrics of all the experiments on a dataset for a given task. The experiment list is defined from MODELS,
    preprocessors and ratios lists.

    Args:
        dataset (str): dataset name using our convention.
        MODELS (list of str): list of model names.
        preprocessors (list of str): list of preprocessor names (None, 'sampler', 'features_sampler').
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        ratios_nf (float): list of nodes features ratios corresponding to
        percentage of modified nodes features.
        task (str): task full name (None, link_prediction, node_classfication)
        metrics_to_save (dict): defines the metrics to gather, format: key is the name used in the
        final dataframe and value corresponds to the name in json metrics files
        output_path (str): path where experiment results are saved.
        baseline (bool): whether to use baseline.
        reduce (str): when multiple repetitions have been performed, aggregate them with specified 
        method.

    Returns:
        pd.DataFrame: gathered experiments results for one dataset on a given task.
    """
    # Output dataset formating
    if ratios_nf is not None:
        if 'sampler' in preprocessors:
            prepro = preprocessors.copy()
            prepro.remove('sampler')
            exps_it = list(it.product(['sampler'], ratios)) + list(it.product(prepro, ratios, ratios_nf))
        else :
            exps_it = list(it.product(preprocessors, ratios, ratios_nf))
        ind_names = ['preprocessor', 'ratio', 'ratio_nf']
    else:
        exps_it = it.product(preprocessors, ratios)
        ind_names = ['preprocessor', 'ratio']
    ind = [('Baseline', 'NA')]+exps_it
    index = pd.MultiIndex.from_tuples(ind, names=ind_names)
    col = [(task,)+ elt for elt in list(it.product(metrics_to_save.keys(), MODELS))]
    columns = pd.MultiIndex.from_tuples(col, names=['task', 'model', 'metric'])
    output_dataset = pd.DataFrame(index=index, columns=columns)
    # Loop to gather metrics
    for MODEL in MODELS:
        if baseline:
            if MODEL in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                metrics = get_exp_metrics(MODEL, None, None, None, None, dataset, output_path, reduce=reduce)
            else:
                metrics = get_exp_metrics(MODEL, task, None, None, None, dataset, output_path, reduce=reduce)
            for M in metrics_to_save.keys():
                if metrics is not None:
                    met = metrics[metrics_to_save[M]]
                    if type(met)==list:
                        output_dataset.loc[('Baseline', 'NA'), (task, M, MODEL)] = metrics[metrics_to_save[M]][0]
                    else:
                        output_dataset.loc[('Baseline', 'NA'), (task, M, MODEL)] = metrics[metrics_to_save[M]]
       
        for preprocessor, ratio, *ratio_nf in exps_it:
            ratio_nf=None if ratio_nf==[] else ratio_nf[0]
            if MODEL in ['gcn', 'gat', 'graphsage'] and task == 'node_classification':
                metrics = get_exp_metrics(MODEL, None, preprocessor, ratio, ratio_nf, dataset, output_path, reduce=reduce)
            else:
                metrics = get_exp_metrics(MODEL, task, preprocessor, ratio, ratio_nf, dataset, output_path, reduce=reduce)
            for M in metrics_to_save.keys():
                if metrics is not None:
                    met = metrics[metrics_to_save[M]]
                    if type(met)==list:
                        output_dataset.loc[(preprocessor,   ratio, ratio_nf), (task, M, MODEL)]  = metrics[metrics_to_save[M]][0]
                    else:
                        output_dataset.loc[(preprocessor,   ratio, ratio_nf), (task, M, MODEL)]  = metrics[metrics_to_save[M]]
    return output_dataset


def summarize_metrics(excel_name, datasets, models_lp, models_nc, preprocessors, ratios, metrics_lp, metrics_nc, output_path, savefig_path, ratios_nf=None, baseline=True, reduce='mean'):
    """Concatenate metrics of all the experiments on several datasets and save them.

    Args:
        excel_name (str): name to save final concatenated metrics dataframe, must end with .xlsx.
        datasets (list of str): list of dataset names using our convention.
        models_lp (list of str): list of model names to use for link_prediction task.
        models_nc (list of str): list of model names to use for node_classification task.       
        preprocessors (list of str): list of preprocessor names (None, 'sampler', 'features_sampler').preprocessors ([type]): [description]
        ratios (list of float): list of ratios corresponding to
        percentages of nodes considered as clients.
        metrics_lp (dict): defines the metrics to gather for link_prediction task, format: key is the name used in the
        final dataframe and value corresponds to the name in json metrics files.
        metrics_nc (dict): defines the metrics to gather for node_classification task, format: key is the name used in the
        finnfal dataframe and value corresponds to the name in json metrics files.
        output_path (str): path where experiment results are saved.
        savefig_path (str): path where figures must be saved.
        ratios_nf (float): list of nodes features ratios corresponding to
        percentage of modified nodes features. Defaults to None.
        baseline (bool, optional): whether to gather experimental baseline metrics. Defaults to True.
        reduce (str): when multiple repetitions have been performed, aggregate them with specified 
        method.

    Returns:
        list pd.DataFrame: gathered experiments results, one df per dataset.
    """
    writer = pd.ExcelWriter(os.path.join(savefig_path, excel_name), engine='xlsxwriter')
    summary = pd.DataFrame([datasets, models_lp, models_nc, preprocessors, ratios, ratios_nf, metrics_lp, metrics_nc, [baseline]]).T
    summary.columns = ['Datasets', 'Models for link pred', 'Models for node classif',
                    'preprocessors', 'ratios', 'ratios_nf', 'Metrics for link pred', 'Metrics for node classif', 'Baseline']
    summary.to_excel(writer, sheet_name='Summary')
    out_datasets_list = []

    for dataset in datasets:
        out_lp = summarize_metrics_dataset_task(dataset, models_lp, preprocessors, ratios, ratios_nf, "link_prediction", metrics_lp, output_path, baseline, reduce)
        out_nc = summarize_metrics_dataset_task(dataset, models_nc, preprocessors, ratios, ratios_nf, "node_classification", metrics_nc, output_path, baseline, reduce)

        out_dataset = pd.concat([out_lp, out_nc], axis=1)
        out_dataset.dropna(axis=0, how='all', inplace=True)
        out_dataset.dropna(axis=1, how='all', inplace=True)
        if len(out_dataset)!=0:
            out_dataset.to_excel(writer, sheet_name=dataset, float_format="%.3f")
            out_datasets_list.append(out_dataset)
    writer.save()
    return out_datasets_list
