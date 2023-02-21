# How to compute the perplexity metric for dexperts (and any other model)

The perplexity metric is computed using the `perplexity-metric.py` script. The script evaluates the perplexity on the WikiText-2 dataset.

Simply launch the script with the following command:

```bash
python perplexity-metric.py --load_json perplexity_args.json --save_json results.json
```

The script will load the configuration file `perplexity_args.json` and save the results in `results.json` under the `perplexity/` folder.
