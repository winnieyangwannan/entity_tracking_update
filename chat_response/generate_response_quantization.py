import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from datasets import Dataset
import tqdm as tqdm
import argparse
from transformer_lens import HookedTransformer
import random

# Inputs
parser = argparse.ArgumentParser()

parser.add_argument('--prompt_path', type=str, default='D:\\Code\\entity_tracking_update\\data')
parser.add_argument('--prompt_name', type=str, default='entity_tracking_3e_2o_1u_prompt_config_3')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf') #meta-llama/Llama-2-7b-chat-hf
parser.add_argument('--hf_token', type=str, default='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')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--save_path', type=str, default='D:\\Code\\entity_tracking_update\\chat_response')
parser.add_argument('--print_response', type=bool, default=True)


args = parser.parse_args()
prompt_path = args.prompt_path
prompt_name = args.prompt_name
model_name = args.model_name
hf_token = args.hf_token
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens
save_path = args.save_path
print_response = args.print_response

seed = 0
random.seed(seed)


# 1. Load Data
data_file = prompt_path + os.sep + prompt_name + '.jsonl'
print(f"data_file:{data_file}")

prompt_name = data_file.split(os.sep)[-1]
print(f"prpmpt_name:{prompt_name}")
save_name = prompt_name.split(".")[0]
print(f"save_name:{save_name}")
model_save_name = model_name.split('/')[-1]
print(f"prpmpt_name:{model_save_name}")
save_file = save_path + os.sep + save_name + '_' + model_save_name + '.jsonl'
print(f"save_file:{save_file}")


with open(data_file, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

prompts = []
for i in range(len(data)):
    prompt = data[i]["prompt"]
    prompts.append(prompt)

# 2. Load Model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
inference_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token_id = tokenizer.eos_token_id  # eos_token_id=50256

if "gpt" in model_name:
    tokenizer.padding_side = 'left'

# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# inference_dtype = torch.float32
hf_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                token=hf_token,
                                                device_map="auto",
                                                torch_dtype=inference_dtype,
                                                quantization_config=quantization_config) #torch_dtype=inference_dtype,
#quantization_config = quantization_config,
#torch_dtype = inference_dtype
model = HookedTransformer.from_pretrained(model_name,
                                             hf_model=hf_model,
                                             dtype=inference_dtype,
                                             fold_ln=False,
                                             fold_value_biases=False,
                                             center_writing_weights=False,
                                             center_unembed=False,
                                             tokenizer=tokenizer,
                                             )
# 3. Generate Input Dataloader
# From str to tokens
input_tokens = model.to_tokens(prompts)
print(f"input_tokens:{input_tokens.shape}")
# build a dataframe

dataset = Dataset.from_dict(
    {
        "input_ids":input_tokens,
    }).with_format("torch")

# Use dataloader to get data into batches
print(f"Length of dataset: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=batch_size)

# 4. Generate Response
response_all = []
with torch.no_grad():
    for prompt in prompts:
        print(f"prompt:{prompt}")
        # inputs["input_ids"] = inputs["input_ids"].to(hf_model.device)
        # outputs = model(input_ids = inputs["input_ids"]) # next token prediction
        output = model.generate(prompt,
                                max_new_tokens=max_new_tokens,
                                temperature=0,
                                do_sample = False)  # generate
        print(f"output:{output}")
        # for ii in output:
        # response = model.tokenizer.decode(output, skip_special_tokens=True)
        if print_response:
            print(output)
        output_dict = {}
        output_dict["response"] = output
        response_all = np.append(response_all, output_dict)
        del output
        torch.cuda.empty_cache()

# 5. Save Results

with open(save_file,'w') as outfile:
    for entry in response_all:
        json.dump(entry,outfile)
        outfile.write('\n')



print("Done :) ")