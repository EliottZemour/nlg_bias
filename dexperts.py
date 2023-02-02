from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
from typing import Union, List
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DexpertsLogitsWarper,
    LogitsProcessorList,
    set_seed,
)


class DExperts:
    def __init__(
        self,
        base_model: Union[str, Path, AutoModelForCausalLM],
        antiexpert_model: Union[str, Path, AutoModelForCausalLM, None] = None,
        expert_model: Union[str, Path, AutoModelForCausalLM, None] = None,
        tokenizer: str = "gpt2",
        alpha: float = 2.0,
        seed: int = 42,
    ):
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        set_seed(seed)

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(
            self.device
        )
        if antiexpert_model:
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
                antiexpert_model, use_auth_token=True
            ).to(self.device)
        else:
            self.antiexpert = None
        if expert_model:
            self.expert = AutoModelForCausalLM.from_pretrained(
                expert_model, use_auth_token=True
            ).to(self.device)
        else:
            self.expert = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

        self.alpha = alpha
        self.logits_processor = LogitsProcessorList(
            [
                DexpertsLogitsWarper(
                    expert_model=expert_model,
                    anti_expert_model=antiexpert_model,
                    alpha=self.alpha,
                    device=self.device,
                )
            ]
        )

    def __call__(self, prompt: str, alpha: float = None):
        encodings_dict = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(self.device)
        encoded_text = encodings_dict["input_ids"]
        attn_mask = encodings_dict["attention_mask"]
        logits = self._get_logits(encoded_text, alpha=alpha)
        return {
            "logits": logits,
            "perplexity": self._get_perplexity(logits, encoded_text),
            "encoded_text": encoded_text,
        }

    def generate(self, **kwargs):
        return self.base_model.generate(
            logits_processor=self.logits_processor,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    def compute_perplexity(self, prompt: str, alpha: float = None):
        encodings_dict = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(self.device)
        encoded_text = encodings_dict["input_ids"]
        attn_mask = encodings_dict["attention_mask"]
        if alpha is None:
            alpha = self.alpha
        logits = self._get_logits(encoded_text, alpha=alpha)
        return self._get_perplexity(logits, encoded_text)

    def _get_perplexity(self, logits, labels, exp=True):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if exp:
            return torch.exp(loss)
        else:
            return loss

    def forward(self, prompt: str, max_length: int = 20, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        return self(prompt, max_length=max_length, alpha=alpha)

    def _get_logits(self, encodings_dict, alpha=None):
        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()

        if alpha is None:
            alpha = self.alpha

        with torch.no_grad():
            # base model prediction
            base_logits = self.base_model(encodings_dict).logits

            # expert prediction
            if self.expert:
                expert_logits = self.expert(encodings_dict).logits
            else:
                expert_logits = base_logits

            # antiexpert prediction
            if self.antiexpert:
                antiexpert_logits = self.antiexpert(encodings_dict).logits
            else:
                antiexpert_logits = base_logits

            if self.antiexpert is not None or self.expert is not None:
                ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
            else:
                ensemble_logits = base_logits

        return ensemble_logits
