import os
import torch
import numpy as np
import random
import argparse
import json

#
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--num_prompt', type=int, default=100)
parser.add_argument('--num_entity', type=int, default=3)
parser.add_argument('--num_object', type=int, default=2)
parser.add_argument('--num_update', type=int, default=1)
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--save_path', type=str, default='D:/Data/entity_tracking_update/prompt')
parser.add_argument('--prompt_config', type=str, default='prompt_config_3')

args = parser.parse_args()
seed = args.seed
num_prompt = args.num_prompt
num_entity = args.num_entity
num_object = args.num_object
num_update = args.num_update
max_new_tokens = args.max_new_tokens
save_path = args.save_path
prompt_config = args.prompt_config

#
torch.set_grad_enabled(False)

#
random.seed(seed)

# object_list = ["book", "toy", "bag", "phone", "cat", "dog", "mouse", "cup", "pen", "coat", "hat", "pillow", "blanket",
#                "mirror", "monitor", "lamp", "key"]
#
# entity_list = ["Box A", "Box B", "Box C", "Box D", "Box E", "Box F", "Box G"]
#
# name_list = ["Amy", "John", "Mary", "Peter", "Anna", "Mike", "Rachel", "James", "Ava", "Emily", "Mia", "Emma", "Sofia",
#              "Leo", "Dylan", "Luke", "Jack"]

# instruction = "Given the description after 'Description:', write a true statement about all boxes and their contents after 'Statement:'. Make sure to keep track of the changes. Update the contents of the boxes according to the changes.\n\n"
config_file = save_path + os.sep + prompt_config+".json"
f = open(config_file)
config = json.load(f)
object_list = config["object_list"]
entity_list = config["entity_list"]
name_list = config["name_list"]
instruction = config["instruction"]
statement = config["statement"]
def generate_one_full_prompt(instruction, entity_list, object_list, name_list, num_entity, num_object, num_update, statement):
    def generate_state_description(num_entity, num_object, entity_list, object_list, instruction):
        # pick x objects form all possible objects
        objects = random.sample(object_list, num_object)
        entity_list = entity_list[0:num_entity]
        # if one of the box is empty
        if num_entity - num_object == 1:
            objects_all = ["nothing"]  # one box contians nothing
            objects_all = np.append(objects_all, objects)

        # shuffle the entity-object relationship and add "the"
        shuffle_num = np.random.permutation(3)  # reorder the objects
        objects_all_shuffle = []
        entity_with_object = {}
        entity_without_object = {}
        for ii in np.arange(num_entity):
            entity_object = objects_all[shuffle_num[ii]]
            if entity_object != "nothing":
                entity_object = "the " + entity_object
                entity_with_object[f'{entity_list[ii]}'] = entity_object
            else:
                entity_without_object[f'{entity_list[ii]}'] = "nothing"
            objects_all_shuffle = np.append(objects_all_shuffle, entity_object)
        # print(f"objects_all_shuffle: {objects_all_shuffle}")
        # print(f"entity_list: {entity_list}")

        # generate state description of all entity-object pairs
        state_description_all = instruction
        for ii in np.arange(num_entity):
            description = entity_list[ii] + " contains " + objects_all_shuffle[ii] + "."
            if ii == 0:
                state_description_all = state_description_all + "Description: " + description
            else:
                state_description_all = state_description_all + " " + description
        return state_description_all, entity_with_object, entity_without_object

    def generate_state_update(num_update, name_list, entity_with_object, entity_without_object, state_description_all):
        state_description_all_update = state_description_all
        for update in np.arange(num_update):
            name = random.sample(name_list, 1)[0]
            shuffle_num = np.random.permutation(2)

            # one update
            move_from = list(entity_with_object.keys())[shuffle_num[0]]
            move_object = list(entity_with_object.values())[shuffle_num[0]]
            move_to = list(entity_without_object.keys())[0]

            state_update_template = f"{name} moves {move_object} from {move_from} to {move_to}."
            state_description_all_update = state_description_all_update + " " + state_update_template

        # no update
        no_update_entity = list(entity_with_object.keys())[shuffle_num[1]]
        state_description_all_update = state_description_all_update + f" {no_update_entity} has no change in its content.\n\n"
        return state_description_all_update

    def generate_statement(state_description_all_update,statement):
        full_prompt = state_description_all_update + statement
        return full_prompt

    # 1. Generate Description
    state_description_all, entity_with_object, entity_without_object = generate_state_description(num_entity,
                                                                                                  num_object,
                                                                                                  entity_list,
                                                                                                  object_list,
                                                                                                  instruction)

    # 2. Generate State Update
    state_description_all_update = generate_state_update(num_update, name_list, entity_with_object,
                                                         entity_without_object, state_description_all)

    # 3. Generate Statement
    full_prompt = generate_statement(state_description_all_update,statement)

    return full_prompt






# generate prompt
dataset = []
for pp in np.arange(num_prompt):
    entry = {}
    entry["prompt"] = generate_one_full_prompt(instruction,entity_list,object_list, name_list, num_entity, num_object, num_update, statement)
    dataset = np.append(dataset,entry)

# save dataset
save_name = 'entity_tracking_' + str(num_entity) + "e_" + str(num_object) + 'o_' + str(num_update)  + 'u_' + prompt_config
data_file = save_path + os.sep + save_name + '.jsonl'
with open(data_file, 'w') as outfile:
    for entry in dataset:
        json.dump(entry, outfile)
        outfile.write('\n')
