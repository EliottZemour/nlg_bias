# How to perform global bias evaluation using DExperts

This tutorial shows how to perform global bias evaluation using DExperts.

## 1. Running evaluation script on prompts (BOLD)
First, the evaluation script can be found in the evaluate_scripts folder: [`evaluate_scripts/evaluateBias_dexperts.py`](evaluate_scripts/evaluateBias_dexperts.py)

The configuration file for the evaluation can be found in the same folder: [`evaluate_scripts/options_dexperts.py`](evaluate_scripts/options_dexperts.py).
Some parameters can be overwritten from the command line.

Here is an example of how to run the evaluation script, while overridding the base model and the output directory:

```bash
python evaluateBias_dexperts.py --base_model gpt2-medium --out_dir results/dexperts_gpt2_med_antionly_temp1_alpha2
```

All the results will be stored in the output directory (`results/dexperts_gpt2_med_antionly_temp1_alpha2` in this example).

## 2. Aggregating results

Now, in order to aggregate the results, we can use the [`evaluate_scripts/global-bias-dexperts.ipynb`](evaluate_scripts/global-bias-dexperts.ipynb) notebook.
In particular, the notebook will compute the global bias tables for any results directory, as following:
    
```python
import os
from colorama import Fore, Back, Style # not necessary, but useful for printing
from utils import avg_diff, get_results


directories = [
    'results/new_temperature/dexperts-base/dexperts_gpt2_expertonly_alpha1/',
    'results/new_temperature/dexperts-medium/dexperts_gpt2_med_expertonly_alpha1/',
]

tables = []
for score_dir in directories:
    score_dir = os.path.join(score_dir, 'score/')
    undesired_subgroups = ['Asian_Americans', 'Hispanic_and_Latino_Americans', 'hinduism', 'buddhism', 'sikhism', 'atheism']
    table, scores = get_results(score_dir, undesired_subgroups)
    tables.append(table)
    print(f"{Fore.LIGHTRED_EX}{score_dir.split('/')[-3].upper()}{Style.RESET_ALL}")
    print(table.get_string(sort_key=operator.itemgetter(1, 0), sortby="Domain"), '\n')

```

The output will be:


NB: I want to draw the attention on the fact that I specify `undesired_subgroups` when computing the results. This is because these demographic subgroups are unfortunately not present in the local bias evaluation (Stereoset), and therefore we do not compute the global bias for them. 