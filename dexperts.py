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
        mode: str = "linear"
    ):
        token = "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"
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
                antiexpert_model, use_auth_token=token
            ).to(self.device)
        else:
            self.antiexpert = None
        if expert_model:
            self.expert = AutoModelForCausalLM.from_pretrained(
                expert_model, use_auth_token=token
            ).to(self.device)
        else:
            self.expert = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

        self.alpha = alpha
        self.mode = mode
        self.logits_processor = LogitsProcessorList(
            [
                DexpertsLogitsWarper(
                    expert_model=expert_model,
                    anti_expert_model=antiexpert_model,
                    alpha=self.alpha,
                    device=self.device,
                    mode=self.mode
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
            
            #pre-softmax 
            if self.antiexpert is not None or self.expert is not None:
                if self.mode == "linear":
                    ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)
                elif self.mode == "bayes":                   
                    combined = torch.stack((expert_logits, antiexpert_logits), 3)
                    combined = torch.logsumexp(combined, 3) # p_expert + p_anti
                    logp_desired_t = expert_logits - combined        
                    ensemble_logits = base_logits + logp_desired_t
                    
            else:
                ensemble_logits = base_logits

        return ensemble_logits

    
#       gedi_outputs = gedi_model(**inputs)
#                 if gedi_past is None:
#                     if gedi_outputs[0].shape[1]>1:
#                         old_logits = torch.log_softmax(gedi_outputs[0][:, :-1, :],-1)

#                         shift_logits = gedi_outputs[0][..., :-1, :].contiguous()
#                         shift_labels = seq_batched[..., 1:].contiguous()
#                         loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#                         logits_r  = -1*loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#                         logits_r = logits_r.view(seq_batched.shape[0], -1)

#                         seq_len = logits_r.shape[1]

#                         logits_r = torch.sum(logits_r,1)


#                         logits_pos,logits_neg = torch.split(logits_r/seq_len,input_ids.shape[0])


#                         logits0 = torch.stack((logits_pos,logits_neg),1)

#                         if "logit_scale" in dir(gedi_model):
#                             logits0 = gedi_model.logit_scale*logits0

#                         if "bias" in dir(gedi_model):
#                             logits0 = logits0 + gedi_model.bias
#                         if not (class_bias==0):
#                             logits0[:,0] += class_bias


#                         logp_desired = torch.log_softmax(logits0,-1)[:,0]
#                         logp_undesired = torch.log_softmax(logits0,-1)[:,1]
#                     else:
#                         seq_len=0
#                         logp_desired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
#                         logp_undesired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
#                         logits_r = torch.zeros(input_ids.shape[0]*2).to(input_ids.device)


#                 seq_len= seq_len+1
#                 gedi_logits= (torch.log_softmax(gedi_outputs[0][:, -1, :],-1)+logits_r.unsqueeze(1))

#                 logits_pos,logits_neg = torch.split(gedi_logits/seq_len,input_ids.shape[0])
#                 logits = torch.stack((logits_pos,logits_neg),2)
#                 if "logit_scale" in dir(gedi_model):
#                     logits = gedi_model.logit_scale*logits

#                 if "bias" in dir(gedi_model):
#                     logits = logits + gedi_model.bias

#                 if not class_bias == 0:
#                     logits[:,:,0] += class_bias

#                 logp_desired_t = torch.log_softmax(logits,-1)[:,:,0]
#                 logp_undesired_t = torch.log_softmax(logits,-1)[:,:,1]

#                 next_token_logits = torch.log_softmax(1*next_token_logits,-1) + disc_weight*(logp_desired_t) #+delta_capped82058721