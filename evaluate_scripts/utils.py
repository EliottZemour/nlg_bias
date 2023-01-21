import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from prettytable import PrettyTable
import operator
from colorama import Fore, Back, Style

def avg_diff(values):
    # return the average pairwise different between all the pairs in the list
    pairs = [(a, b) for idx, a in enumerate(values) for b in values[idx + 1:]]
    difference = list(map(lambda x: abs(x[0] - x[1]), pairs))
    return sum(difference)/len(difference)

def get_results(score_dir, undesired_subgroups):
    path_to_dir = os.path.relpath(score_dir)
    csv_files = [pos_csv for pos_csv in os.listdir(path_to_dir) if pos_csv.endswith('.csv')]
    df = pd.concat([pd.read_csv(os.path.join(path_to_dir, file)) for file in csv_files])
    df = df[~df['subgroup'].isin(undesired_subgroups)]

    scores = []
    groups = df['group'].unique().tolist()
    table = PrettyTable(["Domain", "Model", "Positive", "Neutral", "Negative", "Average", "Toxicity Ratio"])
    # toxicityTable = PrettyTable(["Domain", "Model", "Ratio"])

    for group in groups:
        # group = 'political_ideology'
        data = df[df['group'] == group]
        model_name = data[data['metric'] == 'regard-positive']['model'].values.tolist()
        dir_name = score_dir.split('/')[-3]
        subgroup = data[data['metric'] ==
                        'regard-positive']['subgroup'].values.tolist()
        labels = [model_name[i] + '\n' + subgroup[i]
                for i in range(len(model_name))]

        model_name = data[data['metric'] ==
                        'regard-positive']['model'].unique().tolist()
        subgroup = data[data['metric'] ==
                        'regard-positive']['subgroup'].unique().tolist()

        positive_regards = np.array(
            data[data['metric'] == 'regard-positive']['score'].values.tolist())
        negative_regards = np.array(
            data[data['metric'] == 'regard-negative']['score'].values.tolist())
        neutral_regards = np.array(
            data[data['metric'] == 'regard-neutral']['score'].values.tolist())
        toxicity = np.array(
            data[data['metric'] == 'toxicity-ratio']['score'].values.tolist())

        n_subgroups = len(subgroup)
        for i in range(len(model_name)):
            start_ind, end_ind = n_subgroups * i, n_subgroups * (i+1)
            positive = positive_regards[start_ind:end_ind]
            negative = negative_regards[start_ind:end_ind]
            neutral = neutral_regards[start_ind:end_ind]
            toxic = toxicity[start_ind:end_ind]

            table.add_row(
                [group, dir_name, round(avg_diff(positive), 4), round(
                avg_diff(neutral), 4), round(avg_diff(negative), 4), round(np.mean([avg_diff(positive),
                avg_diff(neutral),avg_diff(negative)]), 4), round(avg_diff(toxic), 4)])

            scores.append({'model': model_name[i], 'group': group,  'positive': round(avg_diff(positive), 4),
                        'negative': round(avg_diff(negative), 4), 'neutral': round(avg_diff(neutral), 4), 'toxicity_ratio': round(avg_diff(toxic), 4)})
    
    return table, scores