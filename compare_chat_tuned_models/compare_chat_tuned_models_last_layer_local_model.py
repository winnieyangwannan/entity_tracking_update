import os
import torch
import numpy as np
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens import utils

parser = argparse.ArgumentParser()
parser.add_argument('--chat_response_file', type=str,
                    default='D:/Code/entity_tracking_update/chat_response/entity_tracking_3e_2o_1u_prompt_config_3_Llama-2-7b-chat-hf.jsonl')
parser.add_argument('--prompt_file', type=str,
                    default='D:/Code/entity_tracking_update/data/entity_tracking_3e_2o_1u_prompt_config_3.jsonl')
parser.add_argument('--model_name_base', type=str,
                    default='meta-llama/Llama-2-7b-hf')  # stanford-crfm/alias-gpt2-small-x21
parser.add_argument('--model_name_tuned', type=str,
                    default='meta-llama/Llama-2-7b-chat-hf')  # meta-llama/Llama-2-7b-chat-hf
parser.add_argument('--model_path', type=str, default='D://Data/Llama/Llama_2/')
parser.add_argument('--save_path', type=str, default='D://Code/entity_tracking_update/compare_chat_tuned_models')

args = parser.parse_args()
chat_response_file = args.chat_response_file
prompt_file = args.prompt_file
model_name_base = args.model_name_base
model_name_tuned = args.model_name_tuned
model_path = args.model_path
save_path = args.save_path

#
torch.set_grad_enabled(False)

# 1. Load Chat Response
print(f"chat_response_file: {chat_response_file}")
with open(chat_response_file, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

responses = []
for i in range(len(data)):
    response = data[i]["response"]
    responses.append(response)

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_tuned)

# 3. Get length of the prompt
print(f"prompt_file: {prompt_file}")

with open(prompt_file, encoding="utf-8") as f:
    prompts_str = [json.loads(line) for line in f]

for i in range(len(prompts_str)):
    prompt = prompts_str[i]["prompt"]
    input_tokens = tokenizer(prompt, return_tensors="pt")
    tokens = torch.flatten(input_tokens['input_ids'])
prompt_length = len(tokens)

# 4. Get Logit of the Last Layer
name2 = model_name_base.split('/')[-1]
model_path_base = model_path + name2
print(f'model_path_base: {model_path_base}')

name2 = model_name_tuned.split('/')[-1]
model_path_tuned = model_path + name2
print(f'model_path_tuned: {model_path_tuned}')


class ModelHelper:
    def __init__(self, path, model_name, device=None, load_in_8bit=False):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print("loading ...")
        print(f"model_path: {path}")
        # print(f"token: {token}")
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        hf_model = AutoModelForCausalLM.from_pretrained(path,
                                                        device_map='cpu')
        self.model = HookedTransformer.from_pretrained(model_name,
                                                       hf_model=hf_model,
                                                       fold_ln=False,
                                                       fold_value_biases=False,
                                                       center_writing_weights=False,
                                                       center_unembed=False,
                                                       tokenizer=self.tokenizer,
                                                       device=self.device)

        print("Done loading")
        # print(self.model.cfg.n_layers)
        self.device = next(self.model.parameters()).device
        self.d_vocab = self.model.cfg.d_vocab
        self.n_layers = self.model.cfg.n_layers

        del hf_model
        torch.cuda.empty_cache()

    def logits_all_layers(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        # Get residual output for each layer
        z_name_filter = lambda name: name.endswith("resid_post")
        self.model.reset_hooks()
        _, cache = self.model.run_with_cache(
            inputs["input_ids"],
            names_filter=z_name_filter,
            return_type=None
        )

        layer_logit_all = torch.zeros(self.n_layers, seq_len, self.d_vocab)
        for layer in range(self.model.cfg.n_layers):
            resid_ln = self.model.ln_final(cache[f'blocks.{layer}.hook_resid_post'])
            layer_logit = self.model.unembed(resid_ln)
            layer_logit_all[layer, :, :] = layer_logit
        return layer_logit_all

    def logits_last_layers(self, text):
        # logits=**Unembed**(**LayerNorm**(**final_residual_stream**))
        # from text to tokens
        inputs = self.tokenizer(text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        # Get residual output for the last layer
        # get name filter for the residial of last layer
        resid_post_hook_name = utils.get_act_name("resid_post", self.model.cfg.n_layers - 1)
        resid_post_name_filter = lambda name: name == resid_post_hook_name
        # run with hook
        self.model.reset_hooks()
        _, cache = self.model.run_with_cache(
            inputs["input_ids"],
            names_filter=resid_post_name_filter,
            return_type=None
        )
        # layer norm
        resid_ln = self.model.ln_final(cache[f'blocks.{self.model.cfg.n_layers - 1}.hook_resid_post'])
        # unembed
        layer_logit = self.model.unembed(resid_ln)
        return layer_logit


model_base = ModelHelper(model_path_base, model_name_base, load_in_8bit=False)

model_tuned = ModelHelper(model_path_tuned, model_name_tuned, load_in_8bit=False)


# 5. Compare the Result of the Last Layer
def compre_two_model_last_layer(response):
    # 1. get last layer logit
    logits_last_layer_base = model_base.logits_last_layers(response)
    logits_last_layer_tuned = model_tuned.logits_last_layers(response)

    # From logit to softmax prbability distribution
    prob_base = logits_last_layer_base.softmax(dim=-1)
    prob_tuned = logits_last_layer_tuned.softmax(dim=-1)

    prob_base = torch.squeeze(prob_base)
    prob_tuned = torch.squeeze(prob_tuned)

    # 2. get the KL divergence
    def KL(P, Q):
        """ Epsilon is used here to avoid conditional code for
        checking that neither P nor Q is equal to 0. """
        epsilon = 0.00001
        P = P + epsilon
        Q = Q + epsilon
        divergence = np.sum(P * np.log(P / Q))
        return divergence

    # For each position in the sequence
    kl_all = []
    for ii in range(len(prob_base)):
        kl = KL(prob_base[ii, :].to('cpu').numpy(), prob_tuned[ii, :].to('cpu').numpy())
        kl_all = np.append(kl_all, kl)

    # 3. Get probability of the answer tokens
    input_tokens = tokenizer(response, return_tensors="pt")
    tokens = torch.flatten(input_tokens['input_ids'])
    tokens = tokens.to(prob_base.device)

    prob_answer_tokens_base = prob_base[:, :-1].gather(dim=-1, index=tokens[1:].unsqueeze(-1)).squeeze(-1)
    prob_answer_tokens_tuned = prob_tuned[:, :-1].gather(dim=-1, index=tokens[1:].unsqueeze(-1)).squeeze(-1)

    prob_answer_tokens_base = prob_answer_tokens_base.to("cpu").numpy()
    prob_answer_tokens_tuned = prob_answer_tokens_tuned.to("cpu").numpy()

    # 4. Get the rank of the answer token
    def answer_tokens_ranks(tokens, probs):
        answer_ranks = []
        for index in range(1, len(tokens)):
            answer_token = tokens[index]
            token_prob = probs[index - 1]
            sorted_token_probs, sorted_token_values = token_prob.sort(descending=True)
            correct_rank = torch.arange(len(sorted_token_values))[
                (sorted_token_values == answer_token).cpu()
            ].item()
            answer_ranks.append(correct_rank)
        return answer_ranks

    answer_ranks_base = answer_tokens_ranks(tokens, prob_base)
    answer_ranks_tuned = answer_tokens_ranks(tokens, prob_tuned)

    return kl_all[prompt_length:], prob_answer_tokens_base[prompt_length:], prob_answer_tokens_tuned[
                                                                            prompt_length:], answer_ranks_base[
                                                                                             prompt_length:], answer_ranks_tuned[
                                                                                                              prompt_length:]


compare_result = []
for response in responses:
    kl, prob_answer_tokens_base, prob_answer_tokens_tuned, rank_answer_tokens_base, rank_answer_tokens_tuned = compre_two_model_last_layer(
        response)
    entry = {}
    entry["kl"] = kl.tolist()
    entry["prob_answer_tokens_base"] = prob_answer_tokens_base.tolist()
    entry["prob_answer_tokens_tuned"] = prob_answer_tokens_tuned.tolist()
    entry["rank_answer_tokens_base"] = rank_answer_tokens_base
    entry["rank_answer_tokens_tuned"] = rank_answer_tokens_tuned
    compare_result = np.append(compare_result, entry)

print(f"kl{len(kl)}")
print(f"prob_answer_tokens_base{len(prob_answer_tokens_base)}")
print(f"prob_answer_tokens_tuned{len(prob_answer_tokens_tuned)}")
print(f"rank_answer_tokens_base{len(rank_answer_tokens_base)}")
print(f"rank_answer_tokens_tuned{len(rank_answer_tokens_tuned)}")

# 6. Save Result
name1 = chat_response_file.split('.')[0]
name2 = name1.split('/')[-1]
save_name = 'compare_models_' + name2
result_file = save_path + os.sep + save_name + '.jsonl'
with open(result_file, "w") as outfile:
    for entry in compare_result:
        json.dump(entry, outfile)
        outfile.write('\n')

print("Done :) ")
