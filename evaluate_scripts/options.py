import argparse


def evaluate_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", required=True,
                        type=str, help='path to the directory containing prompts to generate from to evaluate on')
    parser.add_argument("--out_dir", default='',
                        type=str, help='path to the directory for the ouput')
    parser.add_argument("--evaluate_only", action='store_true',
                        help='only evaluate on the existing sentences')
    parser.add_argument("--n_sentences", default=5, type=int,
                        help='number of sentences generated per prompt')
    parser.add_argument("--model", default='gpt2', type=str,
                        help='model to generate sentences')
    parser.add_argument("--max_new_tokens", default=15, type=int,
                        help='maximum number of new tokens to generate')
    parser.add_argument("--do_sample", default=True, type=bool,
                        help='do sampling from the model')
    parser.add_argument("--temperature", default=1.0, type=float,
                        help='setting temperature of the model')
    parser.add_argument("--top_p", default=0.9, type=float,
                        help='setting top_p of the model')
    parser.add_argument("--top_k", default=None, type=int,
                        help='setting top_k of the model')
    parser.add_argument("--token", default='',
                        help="token for HuggingFace Model Hub if needed")
    opt = parser.parse_args()
    return opt

# CUDA_VISIBLE_DEVICES=1 nohup python -u evaluateBias_inlp.py --prompt_dir ../prompts/ --out_dir results/gpt2_ainlp > inlp.log &
