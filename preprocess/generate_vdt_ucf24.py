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


if __name__ == '__main__':
    random.seed(42)
    data_path = '../data/UCF24/openworld'

    ucf24_classes = read_class_list(os.path.join(data_path, 'vocab_open.txt'))

    results = {}
    for clsname in ucf24_classes:
        print("\nProcessing action: {}...".format(clsname))
        prompt = f"Generate 16 captions that describe the action '{clsname}'. For example, given the action dance, your output will be like: Dance: A person is dancing on the stage, with the body moving rhythmically to music."
        message = [
            {"role": "system", "content": "You are a useful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(model="gpt-4-0613", messages=message, max_tokens=800, request_timeout=60)  # Adjust max_tokens as needed
        res = response.choices[0]['message']['content'].strip().split('\n')
        res = [re.sub(r'\d+. ', '', cap) for cap in res]
        print(res)
        results[clsname] = res
    
    with open(os.path.join(data_path, "vocab_gpt4.json"), "w") as outfile:
        json.dump(results, outfile, indent=4)
