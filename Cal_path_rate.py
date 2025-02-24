import json
import re

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def process_name(name):
    return change_name(standardize(name))

with open('ToolBench_G3_DFS_gpt-4o-2024-08-06_DRAFT.jsonl', 'r') as file:
    data1 = json.load(file)

with open('dataset/ToolBench/test_data/G3.json', 'r') as file:
    data2 = json.load(file)

matching_count = 0
total_count = len(data1)

for i in range(total_count):
    api_results = data1[i]['execute_log']['api_result_ls']
    relevant_apis = data2[i]['relevant APIs']

    api_names_in_results = set()
    for api_call in api_results:
        for api in api_call:
            tool_name = process_name(api['tool_name'])
            api_name = process_name(api['api_name'])
            api_names_in_results.add((tool_name, api_name))
    
    all_apis_present = all((process_name(tool), process_name(api)) in api_names_in_results for tool, api in relevant_apis)
    
    if all_apis_present:
        matching_count += 1

path_rate = matching_count / total_count

print(f"path rate: {path_rate}")
