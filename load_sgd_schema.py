import argparse
import json
import json
import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from graph_bert_score import get_embeddings
domains = ["Alarm", "Banks", "Buses", "Calendar", "Events", "Flights",
           "Homes", "Hotels", "Media", "Messaging", "Movies",
           "Music", "Payment", "RentalCars", "Restaurants", "RideSharing",
           "Services", "Train", "Travel", "Weather"]
domains = []
domain_select_prompt = f'Determine which domain is considered in the following dialogue situation./n' + \
                       f'Choose exactly one domain from this list: {domains}\n' + \
                       f'Answer with only one word, the selected domain from the list. You have to always select the most probable domain.\n' + \
                       f'——- Example 1: ——–\n' + \
                       f'Customer: I need a cheap place to eat\n' + \
                       f'Assistant: We have several not expensive places available. What food are you interested in?\n' + \
                       f'Customer: Chinese food.\n' + \
                       f'Domain: restaurant\n' + \
                       f'—— Example 2: ——–\n' + \
                       f'Customer: What is the address?\n' + \
                       f'Assistant: It’s 123 Northfolk Road.\n' + \
                       f'Customer: That’s all. I also need a train from London.\n' + \
                       f'Domain: train\n' + \
                       f'———–\n' + \
                       f'Now complete the following example:\n'

def load_and_parse_sgd_dataset(folder_path):
    # 遍历文件夹中的所有文件
    examples = []
    exception_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for dialogue in data:
                try:
                    dialogue_id = dialogue['dialogue_id']
                    turns = dialogue['turns']
                    print(f"Dialogue ID: {dialogue_id}")
                    for turn in turns:
                        speaker = turn['speaker']
                        utterance = turn['utterance']
                        print(f"{speaker}: {utterance}")
                    print("-" * 50)
                    examples.append(dialogue)
                except:
                    exception_number += 1
    return examples


def search_domain(domain_candidates):
    domains = ["Alarm", "Banks", "Buses", "Calendar", "Events", "Flights",
               "Homes", "Hotels", "Media", "Messaging", "Movies",
               "Music", "Payment", "RentalCars", "Restaurants", "RideSharing",
               "Services", "Train", "Travel", "Weather"]
    res_domains = []

    for i in domains:
        for domain_str in domain_candidates:
            if i.lower() in domain_str.lower():
                res_domains.append(i)
    return list(set(res_domains))




def load_sgd_by_domain(folder_path):

    # 遍历文件夹中的所有文件
    domain_examples = {}
    exception_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for dialogue in data:
                    try:
                        service_name = "service_name"
                        if "service_name" in dialogue.keys():
                            exp_domain = search_domain(dialogue["service_name"][0])
                        elif "services" in dialogue.keys():
                            exp_domain = search_domain(dialogue["services"][0])
                        else:
                            exp_domain = search_domain(dialogue["service"][0])
                        if exp_domain is not None:
                            if domain_examples.keys() is None or exp_domain not in domain_examples.keys():
                                domain_examples[exp_domain] = []
                                domain_examples[exp_domain].append({dialogue["dialogue_id"]: dialogue["turns"]})
                            else:
                                domain_examples[exp_domain].append({dialogue["dialogue_id"]: dialogue["turns"]})
                    except:
                        pass
    return domain_examples

def load_schema(new_file_path):
    with open(new_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for domain_scheme in data:
        service_name = domain_scheme["service_name"]
        slots = domain_scheme["slots"]

    return slots


def load_domains(folder_path):
    domains = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for dialogue in data:
                if "services" in dialogue.keys():
                    domains.append(dialogue["services"][0])
                elif "service" in dialogue.keys():
                    domains.append(dialogue["service"][0])
                elif "service_name" in dialogue.keys():
                    domains.append(dialogue["service_name"][0])

    domains = list(set(domains))

    new_domains = []
    for domain in domains:
        if len(domain)>1:
            new_domains.append(domain.split("_")[0])
    new_domains = list(set(new_domains))
    return new_domains

def read_example_turns(turns):
    for example in turns:
        dialogue_id = example['dialogue_id']
        turns = example['turns']
        print(f"Dialogue ID: {dialogue_id}")
        for turn in turns:
            speaker = turn['speaker']
            utterance = turn['utterance']
            print(speaker+":"+utterance+"\n")
        break
