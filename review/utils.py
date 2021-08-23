import torch
import math
import yaml
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import nn
from optim import CosineSchedule, TransformerSchedule


def read_configuration(config_file):
    yaml_loader = yaml.FullLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config_dict = yaml.load(f.read(), Loader=yaml_loader)

    return config_dict


def init_seed(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def init_device(config):
    use_gpu = config["use_gpu"]
    device = torch.device("cuda:" + str(config["gpu_id"]) if torch.cuda.is_available() and use_gpu else "cpu")
    return device


def build_optimizer(parameters, learner, learning_rate, config):
    if learner.lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif learner.lower() == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif learner.lower() == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=learning_rate)
    elif learner.lower() == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    elif learner.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=learning_rate)
    elif learner.lower() == 'cosine_warmup':
        optimizer = CosineSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["warmup_steps"], config["training_steps"]
        )
    elif learner.lower() == 'transformer_warmup':
        optimizer = TransformerSchedule(
            optim.AdamW(parameters, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01),
            learning_rate, config["token_dim"], config["warmup_steps"]
        )
    else:
        raise ValueError('Received unrecognized optimizer {}.'.format(learner))
    return optimizer


def move_to_cuda(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_cuda(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_cuda(x, device) for x in maybe_tensor]
    else:
        return maybe_tensor


def topk_sampling(logits, temperature=1.0, top_k=0, top_p=0.9):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        values = torch.topk(logits, top_k)[0]  # B x top_k
        batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
        logits = torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

    if 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)

        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < top_p

        # First mask must always be pickable
        mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        logits = torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)

    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities.squeeze(1)
    token_idx = torch.multinomial(probabilities, 1)

    return token_idx


def greedy_search(logits):
    return logits.squeeze(1).argmax(dim=-1)


class Beam_Search_Hypothesis(object):
    def __init__(self, beam_size, sos_token_idx, eos_token_idx, device, idx2token):
        self.beam_size = beam_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.device = device
        self.idx2token = idx2token

        self.hypthetic_token_idx = [[sos_token_idx]]
        self.completed_hypotheses = []
        self.hyp_scores = torch.zeros(1).to(device)

    def generate(self):
        r""" Pick the hypothesis with max prob among beam_size hypothesises.
        Return:
            List[str]: the generated tokens
        """
        if len(self.completed_hypotheses) == 0:
            generate_idx = self.hypthetic_token_idx[0]
        else:
            generate_idx = max(self.completed_hypotheses, key=lambda hyp: hyp[1])[0]
        generate_tokens = [self.idx2token[idx.item()] for idx in generate_idx]
        return generate_tokens, generate_idx

    def stop(self):
        r""" Determine if the beam search is over.
        Return:
            Bool: ``True`` represents the search over, `Flase` represents the search working.
        """
        return len(self.completed_hypotheses) == self.beam_size

    def step(
        self, gen_idx, token_logits, decoder_states=None, encoder_output=None, encoder_mask=None, input_type='token'
    ):
        r""" A step for beam search.
        Args:
            gen_idx (int): the generated step number.
            token_logits (torch.Tensor): logits distribution, shape: [hyp_num, sequence_length, vocab_size].
            decoder_states (torch.Tensor, optional): the states of decoder needed to choose, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_output (torch.Tensor, optional): the output of encoder needed to copy, shape: [hyp_num, sequence_length, hidden_size], default: None.
            encoder_mask (torch.Tensor, optional): the mask of encoder to copy, shape: [hyp_num, sequence_length], default: None.
        Return:
            torch.Tensor: the next input squence, shape: [hyp_num],
            torch.Tensor, optional: the chosen states of decoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed output of encoder, shape: [new_hyp_num, sequence_length, hidden_size]
            torch.Tensor, optional: the copyed mask of encoder, shape: [new_hyp_num, sequence_length]
        """
        token_probs = F.log_softmax(token_logits, dim=-1).squeeze(1)
        vocab_size = token_probs.shape[-1]

        live_hyp_num = self.beam_size - len(self.completed_hypotheses)
        tmp_hyp_scores = (self.hyp_scores.unsqueeze(1).expand_as(token_probs) + token_probs).view(-1)
        top_scores, top_pos = torch.topk(tmp_hyp_scores, k=live_hyp_num)
        hyp_ids = top_pos // vocab_size
        word_ids = top_pos % vocab_size

        new_hypotheses = []
        new_ids = []
        new_scores = []

        for hyp_id, word_id, score in zip(hyp_ids, word_ids, top_scores):
            new_hyp = self.hypthetic_token_idx[hyp_id] + [word_id]
            if word_id == self.eos_token_idx:
                self.completed_hypotheses.append((new_hyp[1:-1], score / (gen_idx - 1)))
            else:
                new_hypotheses.append(new_hyp)
                new_ids.append(hyp_id)
                new_scores.append(score)

        if len(self.completed_hypotheses) == self.beam_size:
            none_cnt = (decoder_states is not None) + (encoder_output is not None) + (encoder_mask is not None) + 1
            return [None] * none_cnt

        self.hypthetic_token_idx = new_hypotheses
        self.hyp_scores = torch.tensor(new_scores).to(self.device)

        hyp_num = len(self.hypthetic_token_idx)
        if input_type == 'token':
            input_seq = [hyp[-1] for hyp in self.hypthetic_token_idx]
            input_seq = torch.tensor(input_seq).unsqueeze(1).to(self.device)
        elif input_type == 'whole':
            input_seq = torch.tensor(self.hypthetic_token_idx).to(self.device)
        else:
            raise ValueError("The input type must be in ['token', 'whole'].")

        returns = [input_seq]

        if decoder_states is not None:
            new_ids = torch.tensor(new_ids).to(self.device)
            if isinstance(decoder_states, tuple):
                (x, y) = decoder_states
                decoder_states = (x[:, new_ids, :], y[:, new_ids, :])
            else:
                decoder_states = decoder_states[:, new_ids, :]
            returns += [decoder_states]

        if encoder_output is not None:
            encoder_output = encoder_output[0:1].repeat(hyp_num, 1, 1)
            returns += [encoder_output]

        if encoder_mask is not None:
            encoder_mask = encoder_mask[0:1].repeat(hyp_num, 1)
            returns += [encoder_mask]

        return returns
