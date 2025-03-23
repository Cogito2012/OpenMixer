import random
import re
import os
import copy
import json

import openai
openai.api_key = "YOUR_OPENAI_KEY_HERE"


def read_class_list(filepath):
    class_list = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_list.append(line.strip())
    return class_list

def read_class_description(filepath):
    with open(filepath, 'r') as f:
        refine_maps = json.load(f)
    return refine_maps


def run_gpt4(class_name):
    prompt = """
What are the visual features for distinguishing {}? Please describe with a few short sentences.
"""
    cls_name = re.sub("_", " ", class_name)
    message = [
        {"role": "system", "content": "You are a useful assistant."},
        {"role": "user", "content": prompt.format(cls_name)}
    ]

    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    max_tokens=1024,
    temperature=1.2,
    messages = message)

    # parse the response
    result = response['choices'][0]['message']['content']
    return result


def generate_different_meaning():
    jhmdb_classes = read_class_list(os.path.join(data_path, 'vocab_open.txt'))

    results = {}
    for clsname in jhmdb_classes:
        print("\nProcessing action: {}...".format(clsname))
        cls_name = re.sub("_", " ", clsname)
        prompt = f"Generate 16 unique sentences describing the action '{cls_name}':"
        message = [
            {"role": "system", "content": "You are a useful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(model="gpt-4-0613", messages=message, max_tokens=800, request_timeout=60)  # Adjust max_tokens as needed
        res = response.choices[0]['message']['content'].strip().split('\n')
        print(res)

        results[clsname] = res
    
    with open(os.path.join(data_path, "vocab_gpt4_m16.json"), "w") as outfile:
        json.dump(results, outfile)


def generate_same_meaning():
    class_descriptions = read_class_description(os.path.join(data_path, 'vocab_gpt3.5.json'))

    results = {}
    for clsname, desc in class_descriptions.items():
        print("\nProcessing action: {}...".format(clsname))
        cls_name = re.sub("_", " ", clsname)
        cap_prefix, cap = desc.split(": ")
        prompt = f"Given a sport action type from JHMDB dataset, such as '{cls_name}, please provide 16 different sentences that express the same meaning of the caption: '{cap}'."
        message = [
            {"role": "system", "content": "You are a useful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(model="gpt-4-0613", messages=message, max_tokens=800, request_timeout=60)  # Adjust max_tokens as needed
        res = response.choices[0]['message']['content'].strip().split('\n')
        res = [desc] + [re.sub(r'\d+.', f'{cap_prefix}:', cap) for cap in res]
        print(res)

        results[clsname] = res
    

    with open(os.path.join(data_path, "vocab_gpt4_m16new.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == '__main__':
    random.seed(42)
    data_path = '../data/JHMDB/openworld'
    
    # generate_different_meaning()

    # generate_same_meaning()

    class_descriptions = read_class_description(os.path.join(data_path, 'vocab_gpt3.5.json'))

    # get candidate verbs
    seen_classes = read_class_list(os.path.join(data_path, 'train50%', 'vocab_closed_0.txt'))
    verbs_list = [clsname.split("_")[0] for clsname in seen_classes]
    
    prompt = """In this task, you are given an input sentence. 
Your job is to tell me 16 output sentences with different meanings by only changing the action verbs using a list of candidate verbs. 
The output format should be a dictionary of key-value pair where keys are the verbs you are choosing, and values are the generated sentences."""

    results = {}
    for clsname, desc in class_descriptions.items():
        if clsname not in seen_classes:
            continue  # only process the seen classes
        print("\nProcessing action: {}...".format(clsname))
        cls_name = re.sub("_", " ", clsname)
        cap_prefix, cap = desc.split(": ")
        verbs_sub = copy.deepcopy(verbs_list)
        verbs_sub.remove(clsname.split("_")[0])
        verbs_sub = ', '.join(verbs_sub)
        message = [
            {"role": "system", "content": "You are a useful assistant."},
            {"role": "user", "content": prompt + f" The input sentence: {cap} The candidate verb list: [{verbs_sub}]."}
        ]
        response = openai.ChatCompletion.create(model="gpt-4-0613", messages=message, max_tokens=800, request_timeout=60)
        res = response.choices[0]['message']['content'].strip().split('\n')
        result_list = []
        for strline in res:
            if ': ' not in strline:
                continue
            strline = re.sub("\"", "", strline.strip(","))
            prefix, sentence = strline.split(': ')
            result_list.append("{}: {}".format(prefix.capitalize(), sentence))
            if len(result_list) == 8:
                break
        print(result_list)
        
        results[clsname] = result_list

    with open(os.path.join(data_path, 'train50%', "hardneg_closed_0.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)

    