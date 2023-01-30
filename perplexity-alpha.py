import argparse
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset
from dexperts import DExperts

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_json',
        help='Save settings to file in json format. Ignored in json file', default=None)
    parser.add_argument('--load_json',
        help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    stride = args.stride
    alpha = args.dexperts['alpha']
    max_length = args.max_length
    max_length_pattern = args.max_length_pattern

    # instantiate dexperts
    dexperts = DExperts(
        base_model=args.base_model,
        expert_model=args.dexperts.get('expert_model', None),
        antiexpert_model=args.dexperts.get('antiexpert_model', None),
        tokenizer=args.base_model,
        alpha=alpha,
    )
    device = dexperts.device

    # set up parameters
    if 'bloom' in args.base_model:
        max_length = (max_length if max_length > 0 else 2048) - max_length_pattern
    else:
        max_length = (max_length if max_length > 0 else dexperts.base_model.config.n_positions) - max_length_pattern
    if stride <= 0:
        stride = max_length

    # load dataset and tokenize
    data = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    encodings = dexperts.tokenizer('\n\n'.join(data['text']), return_tensors='pt')

    ppls = {}
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        # compute perplexity
        lls_regular = []
        ppl_regular = None

        for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
            # if i> 1:
            #     break
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                # loss_regular = compute_loss(input_ids, labels=target_ids)
                logits = dexperts._get_logits(input_ids, alpha=alpha)
                # print(logits.shape)
                loss_regular = dexperts._get_perplexity(logits=logits, labels=target_ids, exp=False)
                # print(loss_regular)
                log_likelihood_regular = loss_regular * trg_len

            lls_regular.append(log_likelihood_regular)

            ppl_regular = torch.exp(torch.stack(lls_regular).sum() / end_loc)
            # print(f'Perplexity after {i} tokens: {ppl_debiased} (debiased) vs {ppl_regular} (regular)')
        print(f'Final perplexity: {ppl_regular}')
        ppls[alpha] = ppl_regular.item()
    
    print(f'Final perplexities: {ppls}')

    if args.save_json:
        with open(args.save_json, 'wt') as f:
            args.__dict__['perplexity'] = ppls
            json.dump(args.__dict__, f, indent=2)

if __name__ == '__main__':
    main()
