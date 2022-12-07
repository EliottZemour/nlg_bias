import numpy as np
import torch
from torch.nn import functional as F

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
)

import matplotlib.pyplot as plt
from tqdm import tqdm

def hellinger(p: torch.Tensor, q: torch.Tensor):
    """
    Compute the Hellinger distance between two probability distributions.
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def kl_divergence(p: torch.Tensor, q: torch.Tensor):
    """
    Compute the KL divergence between two probability distributions.
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def local_hellinger(context_a, context_b, model, tokenizer, device):
    """
    Compute the local Hellinger distance between two contexts.
    """
    with torch.no_grad():
        input_ids_a = tokenizer.encode(context_a, return_tensors="pt").to(device)
        input_ids_b = tokenizer.encode(context_b, return_tensors="pt").to(device)

        logits_a = model(input_ids_a)[0]
        logits_b = model(input_ids_b)[0]

        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        return hellinger(probs_a[0, -1].cpu().numpy(), probs_b[0, -1].cpu().numpy())

def local_kl_divergence(context_a, context_b, model, tokenizer, device):
    """
    Compute the local KL divergence between two contexts.
    """
    with torch.no_grad():
        input_ids_a = tokenizer.encode(context_a, return_tensors="pt").to(device)
        input_ids_b = tokenizer.encode(context_b, return_tensors="pt").to(device)

        logits_a = model(input_ids_a)[0]
        logits_b = model(input_ids_b)[0]

        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        return kl_divergence(probs_a[0, -1].cpu().numpy(), probs_b[0, -1].cpu().numpy())

def local_bias(context_list_a, context_list_b, model, tokenizer, device):
    """
    Compute the local bias between two contexts.
    """
    
    hellinger_distances = []
    kl_divergences = []
    for context_a, context_b in tqdm(zip(context_list_a, context_list_b)):
        hellinger_distances.append(local_hellinger(context_a, context_b, model, tokenizer, device))
        kl_divergences.append(local_kl_divergence(context_a, context_b, model, tokenizer, device))

    return {
        "hellinger": np.mean(hellinger_distances),
        "kl_divergence": np.mean(kl_divergences),
    }

def local_hellinger_dexperts(context_a, context_b, dexperts, alpha, device):
    """
    Compute the local Hellinger distance between two contexts.
    """
    with torch.no_grad():
        logits_a = dexperts(context_a, alpha=alpha)['logits']
        logits_b = dexperts(context_b, alpha=alpha)['logits']

        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        return hellinger(probs_a[0, -1].cpu().numpy(), probs_b[0, -1].cpu().numpy())

def local_kl_divergence_dexperts(context_a, context_b, dexperts, alpha, device):
    """
    Compute the local KL divergence between two contexts.
    """
    with torch.no_grad():
        logits_a = dexperts(context_a, alpha=alpha)['logits']
        logits_b = dexperts(context_b, alpha=alpha)['logits']

        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)

        return kl_divergence(probs_a[0, -1].cpu().numpy(), probs_b[0, -1].cpu().numpy())


def local_bias_dexperts(context_list_a, context_list_b, dexperts, alpha=2.0, device='cpu'):
    """
    Compute the local bias between two contexts.
    """
    
    hellinger_distances = []
    kl_divergences = []
    for context_a, context_b in tqdm(zip(context_list_a, context_list_b)):
        hellinger_distances.append(local_hellinger_dexperts(context_a, context_b, dexperts, alpha, device))
        kl_divergences.append(local_kl_divergence_dexperts(context_a, context_b, dexperts, alpha, device))

    return {
        "hellinger": np.mean(hellinger_distances),
        "kl_divergence": np.mean(kl_divergences),
    }