import openai
import json
import logging
import sys
import argparse
import numpy as np
import requests
import random
import os
import subprocess
import re
import importlib.util
import pickle
from tqdm import tqdm
import time
from requests.models import Response
from pydantic import BaseModel
from typing import Union


openai.api_base = " "
openai.api_key = " "

temperature = 0.2
top_p = 1
max_tokens = 2000
proxies = {}
rapidapi_key = " "

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str

def prepare_tool_name_and_url(tools_root, info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name))
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from {tools_root}.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from {tools_root}.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string

def process_error(response):
    save_cache_flag = False
    switch_flag = False
    if "The request to the API has timed out. Please try again later, or if the issue persists" in str(response):
        return_dict = {"error": "API temporarily not working error...", "response": response}

    if "Your Client (working) ---> Gateway (working) ---> API (not working)" in str(response):
        return_dict = {"error": "API not working error...", "response": response}
        
    elif "Unauthorized" in str(response) or "unauthorized" in str(response):
        save_cache_flag = True
        return_dict = {"error": "Unauthorized error...", "response": response}
    
    elif "You are not subscribed to this API." in str(response):
        switch_flag = True
        return_dict = {"error": "Unsubscribed error...", "response": response}
    
    elif "Too many requests" in str(response):
        switch_flag = True
        return_dict = {"error": "Too many requests error...", "response": response}

    elif "You have exceeded" in str(response) or "you are being rate limited"  in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Access restricted. Check credits balance or enter the correct API key." in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}
    
    elif "Oops, an error in the gateway has occurred." in str(response):
        switch_flag = True
        return_dict = {"error": "Gateway error...", "response": response}

    elif "Blocked User. Please contact your API provider." in str(response):
        switch_flag = True
        return_dict = {"error": "Blocked error...", "response": response}
    
    elif "error" in str(response):
        return_dict = {"error": "Message error...", "response": response}

    else:
        save_cache_flag = True
        return_dict = {"error": "", "response": response}
    return return_dict, save_cache_flag, switch_flag

def run(toolbench_code_string, toolbench_api_name, toolbench_input_params_str):
    success_flag = False
    switch_flag = False
    save_cache = False
    print(toolbench_code_string)
    try:
        exec(toolbench_code_string)
        eval_func_str = f"{toolbench_api_name}({toolbench_input_params_str})"
        new_func = eval(eval_func_str)
        response, save_cache, switch_flag = process_error(new_func)
        success_flag = True
    except Exception as e:
        response = {"error": f"Function executing {toolbench_code_string} error...\n{e}", "response": ""}
        save_cache = False
    return success_flag, switch_flag, response, save_cache


def dict_shorten(origin: dict, schema: dict):
    for key, value in list(origin.items()):
        if key not in schema:
            del origin[key]
        else:
            if isinstance(value, dict):
                dict_shorten(value, schema[key]) 
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        for item in value:
                            dict_shorten(item, schema[key][0]) 
    return origin

def observation_shorten(schema_root, response_dict, category, tool_name, api_name, strip_method):
    if strip_method == "filter" or (strip_method == "random" and random.random() > 0.5):
        if isinstance(response_dict["response"], dict):
            if os.path.exists(os.path.join(schema_root, category)):
                if os.path.exists(os.path.join(schema_root, category, tool_name+".json")):
                    schema_dicts = json.load(open(os.path.join(schema_root, category, tool_name+".json"), "r"))
                    api_list = schema_dicts["api_list"]
                    schema = None
                    for schema_dict in api_list:
                        schema_api_name = change_name(standardize(schema_dict["name"]))
                        if schema_api_name == api_name and len(schema_dict["schema"]) > 0:
                            schema = schema_dict["schema"]
                            break
                    if schema is not None:
                        response_dict["response"] = dict_shorten(response_dict["response"], schema)
    return str(response_dict["response"])


def get_rapidapi_response(input_dict: dict, api_customization: bool=False, tools_root: str="data.toolenv.tools", schema_root: str="data/toolenv/response_examples"):
    info = Info
    info.category = input_dict['category']
    info.tool_name = input_dict['tool_name']
    info.api_name = input_dict['api_name']
    info.tool_input = input_dict['tool_input']
    info.strip = input_dict['strip']
    rapidapi_key = input_dict['rapidapi_key']

    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(tools_root, info)
    tool_input = info.tool_input
    
    strip_method = info.strip
    if type(tool_input) == str:
        try:
            tool_input = json.loads(tool_input)
        except Exception as e:
            if tool_input == "":
                tool_input = {}
            else:
                print(f"Can not parse tool input into json: {tool_input}")
                response_dict = {"error": f"Tool input parse error...\n", "response": ""}
                return response_dict
    
    input_params_str = ""
    if len(tool_input) > 0:
        for key, value in tool_input.items():
            if isinstance(value, str):
                input_params_str += f'{key}="{value}", '
            else:
                input_params_str += f'{key}={value}, '
    if not api_customization:
        input_params_str += f"toolbench_rapidapi_key='{rapidapi_key}'"
    success_flag, switch_flag, response_dict, save_cache = run(code_string, api_name, input_params_str)
    observation = observation_shorten(schema_root, response_dict, standard_category, tool_name.replace(f"_for_{standard_category}", ""), api_name, strip_method)
    result = str(observation)[:2048]
    return {"error": response_dict['error'], "response": result}


def build_index(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name not in index:
                index[dir_name] = []
            index[dir_name].append(root)
    return index

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

def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0

def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))

def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data

def openai_response(messages, temperature, top_p, max_tokens, model, is_string): 
    try:
        ans = get_response(messages, temperature, top_p, max_tokens, model)
        print(ans)
        if is_string:
            return ans
        else:
            cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
            ans = json.loads(cleaned_text)
            return ans
    except Exception as e: 
        print(f"Caught an exception of type: {e}") 
        print("pausing")
    time.sleep(2)


def get_response(messages, temperature, top_p, max_tokens, model):

    response = openai.ChatCompletion.create(model=model,
        messages=messages,
        temperature=temperature, 
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if not response.get("error"):
        return response["choices"][0]["message"]["content"]
    return response["error"]["message"]


def choose_tool(question, Tool_dic, tool_used, model_name):
    template = "You are a helpful assistant."
    Tool_list = []
    for ele in Tool_dic:
            if str(ele['ID']) not in tool_used:
                Tool_list.append(str(ele))
    prompt = (
        f"This is the user's question: {question}\n"
        "These are the tools you can select to solve the question:\n"
        "Tool List:\n"
        f"{Tool_list}\n\n"
        "Please note that: \n"
        "1. You should only choose one tool from the Tool List to solve this question.\n"
        "2. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. Do not select from the functions contained in the tool. An example output looks like:\n"
        "'''\n"
        "Example: ```json{\"ID\": XX}```\n"
        "'''\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
    print(result)
    return result

def choose_API(API_instruction, API_list, question, model_name):

    template = "You are a helpful assistant."
    prompt = (
        f"{API_instruction}\n"
        "This is an API Tool instruction. Given a question, you should choose APIs from the API list you want to use for this question in this instruction.\n"
        f"This is the API list: {API_list}\n"
        "Please note that: \n"
        "1. The APIs you choose must in the API list.\n"
        "2. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An example output looks like:\n"
        "```\n"
        "Output_Example: ```json[\"api1\", \"api2\", ...]```\n"
        "```\n"
        f"Question: {question}\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
    print(result)
    return result

def choose_parameter(API_instruction, question, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "Given a user's question, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
        f"This is API tool documentation: {API_instruction}\n"
        "Please note that: \n"
        "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
        "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {\"Parameters\":{}}\n"
        "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
        "4. If you need to use this API multiple times, please set \"Parameters\" to a list.\n"
        "5. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An examples output looks like:\n"
        "'''\n"
        "Example 1: ```json{\"Parameters\":{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}\n```"
        "'''\n"
        f"This is user's question: {question}\n"
        "Output:\n"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    for i in range(3):
        try:
            print(prompt)
            result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
            print(result)
            return result["Parameters"]
        except:
            continue
    return {}

def choose_parameter_depend(API_instruction, question, previous_log, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
        "Please note that: \n"
        "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
        "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {\"Parameters\":{}}\n"
        "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
        "4. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers for your reference.\n"
        "5. If you need to use this API multiple times,, please set \"Parameters\" to a list.\n"
        "6. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An examples output looks like:\n"
        "'''\n"
        "Example 1: ```json{\"Parameters\":{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}\n```"
        "'''\n"
        f"There are logs of previous questions and answers: \n {previous_log}\n"
        f"This is the current user's question: {question}\n"
        f"This is API tool documentation: {API_instruction}\n"
        "Output:\n"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    for i in range(3):
        try:
            print(prompt)
            result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
            print(result)
            return result["Parameters"]
        except:
            continue
    return {}

def answer_generation(question, call_result, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "You should answer the question based on the response output by the API tool."
        "Please note that:\n"
        "1. Answer the question in natural language based on the API response reasonably and effectively.\n"
        "2. The user cannot directly get API response, "
        "so you need to make full use of the response and give the information "
        "in the response that can satisfy the user's question in as much detail as possible.\n"
        f"This is the user's question:\n {question}\n"
        f"This is the API response:\n {call_result}\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=True)
    print(result)
    return result

def answer_generation_depend(question, call_result, model_name, previous_log):
    template = "You are a helpful assistant."
    prompt = (
        "You should answer the question based on the response output by the API tool."
        "Please note that:\n"
        "1. Try to organize the response into a natural language answer.\n"
        "2. We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.\n"
        "3. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.\n"
        f"There are logs of previous questions and answers: \n {previous_log}\n"
        f"This is the user's question: {question}\n"
        f"This is the response output by the API tool: \n{call_result}\n"
        "We will not show the API response to the user, "
        "thus you need to make full use of the response and give the information "
        "in the response that can satisfy the user's question in as much detail as possible.\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=True)
    print(result)
    return result

def answer_check(question, answer, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "Please check whether the response can reasonably and accurately answer the question."
        "If can, please output 'YES'; If not, please output 'NO'\n"
        "You need to give reasons first and then decide whether the response can reasonably and accurately answer the question. "
        "You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. Two example outputs look like:\n"
        "Example 1: ```json{\"Reason\": \"The reason why you think the response can reasonably and accurately answer the question\", \"Choice\": \"Yes\"}```\n"
        "Example 2: ```json{\"Reason\": \"The reason why you think the response cannot reasonably and accurately answer the question\", \"Choice\": \"No\"}```\n"
        f"This is the user's question: {question}\n"
        f"This is the response: {answer}\n"
        "Output: "
    )
    messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
    print(result)
    if 'yes'.lower() in str(result).lower():
        return 1
    else:
        return -1

def retrieval(question, Tool_dic, dataset, data, api_list, api_used, tool_used, ind, model_name, index, previous_log=None):
    tool_id = choose_tool(question, Tool_dic, tool_used, model_name)
    if tool_id == {}:
        return tool_id, "", "", "", ""
    try:
        if str(tool_id["ID"]) not in dataset:
            return tool_id, "", "", "", ""
    except:
            return tool_id, "", "", "", ""
    tool_instruction = dataset[str(tool_id["ID"])]
    API_instruction = str(tool_instruction)
    API_tool = standardize(tool_instruction["tool_name"])
    category = tool_instruction["category"]
    API_list = []
    print("api_used")
    print(api_used)
    for ele in tool_instruction["tool_guidelines"].keys():
        if ele in api_list and ele not in api_used:
            API_list.append(ele)
    
    api_selection = choose_API(API_instruction, API_list, question, model_name)
    api_result = []
    try:
        if len(api_selection) == 0:
            call_result = ""
            print("No Calling")
            return tool_id, api_result, call_result, tool_instruction, API_instruction
    except:
            call_result = ""
            return tool_id, api_result, call_result, tool_instruction, API_instruction

    for api in api_selection:
        if previous_log ==None:
            parameter = choose_parameter(tool_instruction["tool_guidelines"][api], question,
                                            model_name)
        else:
            parameter = choose_parameter_depend(tool_instruction["tool_guidelines"][api],
                                                question, previous_log,
                                                model_name)
        if parameter == -1:
            continue
        api_result.append({"categoty": category, "tool_name": API_tool, "api_name": api, "parameters": parameter})
    if len(api_result) == 0:
        call_result = ""
        return tool_id, api_result, call_result, tool_instruction, API_instruction
    call_results = []
    for api in api_result:
        api_name = change_name(standardize(api["api_name"]))
        parameters = {}
        if isinstance(api["parameters"], dict):
            for key in api["parameters"]:
                value = api["parameters"][key]
                key = change_name(key)
                parameters[key] = value
        elif isinstance(api["parameters"], list):
            for para_ls in api["parameters"]:
                for key in para_ls:
                    value = para_ls[key]
                    key = change_name(key)
                    parameters[key] = value
        payload = {
                "category": api["categoty"],
                "tool_name": api['tool_name'],
                "api_name": api_name,
                "tool_input": parameters,
                "strip": "filter",
                "rapidapi_key": rapidapi_key
            }
        call_result = get_rapidapi_response(payload)

        call_results.append(str(call_result))
    call_result = '\n\n'.join(call_results)
    
    return tool_id, api_result, call_result, tool_instruction, API_instruction


def task_decompose(question, Tool_dic, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "We have the following tools:\n"
        f"{Tool_dic}\n"
        "You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step.\n"
        f"This is the user's question: {question}\n"
        "Please note that: \n"
        "1. You should only decompose this complex user's question into some simple subtasks which can be executed easily by using a single tool.\n"
        "2. Each simple subtask should be expressed into natural language.\n"
        "3. Each subtask should contain the necessary information from the original question and should be complete, explicit and self-consistent.\n"
        "4. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An example output looks like:\n"
        "'''\n"
        "```json{\"Tasks\": [\"Task 1\", \"Task 2\", ...]}```\n"
        "'''\n"
        "Output:"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
    print(result)
    return result


def task_topology(question, task_ls, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "Given a complex user's question, I have decompose this question into some simple subtasks"
        "I think there exists a logical connections and order amontg the tasks. "
        "Thus you need to help me output this logical connections and order.\n"
        "You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An example output looks like:\n"
        "'''\n"
        "```json[{\"task\": task, \"id\", task_id, \"dep\": [dependency_task_id1, dependency_task_id2, ...]}]```\n"
        "'''\n"
        "The \"dep\" field denotes the id of the previous task which generates a new resource upon which the current task depends. If there are no dependencies, set \"dep\" to -1.\n\n"
        f"This is user's question: {question}\n"
        "These are subtasks of this question:\n"
        f"{task_ls}\n"
        "Output: "
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=False)
    print(result)
    for i in range(len(result)):
        if isinstance(result[i]['dep'], str):
            temp = []
            for ele in result[i]['dep'].split(','):
                temp.append(int(ele))
            result[i]['dep'] = temp
        elif isinstance(result[i]['dep'], int):
            result[i]['dep'] = [result[i]['dep']]
        elif isinstance(result[i]['dep'], list):
            temp = []
            for ele in result[i]['dep']:
                temp.append(int(ele))
            result[i]['dep'] = temp
        elif result[i]['dep'] == -1:
            result[i]['dep'] = [-1]
    return result


def answer_summarize(question, answer_task, model_name):
    template = "You are a helpful assistant."
    prompt = (
        "We break down a complex user's problems into simple subtasks and provide answers to each simple subtask. "
        "You need to organize these answers to each subtask and form a self-consistent final answer to the user's question\n"
        f"This is the user's question: {question}\n"
        f"These are subtasks and their answers: {answer_task}\n"
        "Final answer:"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(prompt)
    result = openai_response(messages, temperature, top_p, max_tokens, model_name, is_string=True)
    print(result)
    return result



def task_execution(data_type,
                   base_path, index, dataset, test_data, progress_file,
                   start_index, total_files, retrieval_num, ind, model_name, method):
    with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
        for i, data in enumerate(test_data[start_index:], start=start_index):
            answer_ls = []
            question = data["query"]
            Tool_dic = data["Tool_dic"]
            api_list = []
            for ele in Tool_dic:
                temp_tool = dataset[str(ele['ID'])]
                for api in data['api_list']:
                    for temp_api_name, temp_api in temp_tool['tool_guidelines'].items():
                        if temp_api_name == api['api_name'] and temp_tool['tool_name'] == api['tool_name']:
                            api['api_description'] = temp_api['description']
                            api_list.append(api)
            api_name_list = [api["api_name"] for api in data["api_list"]]
            temp = task_decompose(question, api_list, model_name)['Tasks']
            task_ls = []
            for t in range(len(temp)):
                task_ls.append({"task": temp[t], "id": t + 1})
            task_ls = task_topology(question, task_ls, model_name)
            task_depend = {}
            for task_dic in task_ls:
                task_depend[task_dic['id']] = {'task': task_dic['task'], 'answer': ''}
            answer_task = []
            api_result_ls = []
            call_result_ls = []
            parameter_ls = []
            for task_dic in task_ls:
                task = task_dic['task']
                depend_id = task_dic['dep']
                tool_used = []
                api_used = []
                for r in range(retrieval_num):
                    Tool_list = []
                    for ele in Tool_dic:
                        ele['Description'] = dataset[str(ele['ID'])]['tool_description']
                    for ele in Tool_dic:
                        if str(ele['ID']) not in tool_used:
                            Tool_list.append(str(ele))
                    if Tool_list == []:
                        break
                    if depend_id[0] == -1:
                        tool_id, api_result, call_result, tool_instruction, API_instruction = retrieval(task,
                                                                                                        Tool_dic,
                                                                                                        dataset,
                                                                                                        data,
                                                                                                        api_name_list, 
                                                                                                        api_used,
                                                                                                        tool_used,
                                                                                                        ind,
                                                                                                        model_name,
                                                                                                        index)
                        call_result = str(call_result)
                        answer = answer_generation(task, call_result, model_name)
                    else:
                        previous_log = []
                        for ids in depend_id:
                            previous_log.append(task_depend[ids])
                        tool_id, api_result, call_result, tool_instruction, API_instruction = retrieval(task,
                                                                                                        Tool_dic,
                                                                                                        dataset,
                                                                                                        data,
                                                                                                        api_name_list, 
                                                                                                        api_used,
                                                                                                        tool_used,
                                                                                                        ind,
                                                                                                        model_name,
                                                                                                        index,
                                                                                                        previous_log=previous_log)
                        call_result = str(call_result)
                        answer = answer_generation_depend(task, call_result, model_name,
                                                            previous_log=previous_log)

                    check_index = answer_check(task, answer, model_name)
                    if check_index == 1:
                        if len(api_result) !=0:
                            api_result_ls.append(api_result)
                            call_result_ls.append(call_result)
                        break
                    else:
                        if len(api_result) !=0:
                            api_result_ls.append(api_result)
                            call_result_ls.append(call_result)
                        answer_ls.append({'task': task, 'answer': answer})
                        try:
                            for ele in api_result:
                                api_used.append(str(ele["api_name"]))
                            API_list = []
                            for ele in tool_instruction["tool_guidelines"].keys():
                                if ele in api_name_list and ele not in api_used:
                                    API_list.append(ele)
                            if len(API_list) == 0:
                                tool_used.append(str(tool_id["ID"]))
                        except:
                            continue
                        print('****Try Again****')
                answer_task.append({'task': task, 'answer': answer})
                task_depend[task_dic['id']]['answer'] = answer
            final_answer = answer_summarize(question, answer_task, model_name)
            check_index = answer_check(question, final_answer, model_name)

            ind = ind + 1
            with open(f'''ToolBench_{data_type}_DFS_{model_name}_{method}.jsonl''', 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "ID": ind,
                    "question": question,
                    "final_answer": final_answer,
                    "subtask": task_ls,
                    "answer_subtask": answer_task,
                    "answer_wrong": answer_ls,
                    "check_index": check_index,
                    "execute_log": {
                        "api_result_ls": api_result_ls,
                        "parameter_ls": parameter_ls,
                        "call_result_ls": call_result_ls
                    }
                }, ensure_ascii=False,indent=4)
                f.write(line + '\n')

            print(final_answer)
            update_progress(progress_file, i + 1)
            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o-2024-08-06')
    parser.add_argument('--data_type', type=str, default='G3', help='G2 or G3')
    parser.add_argument('--tool_root_dir', type=str, default='.toolenv/tools/')
    parser.add_argument('--method', type=str, default='Initial', help='Initial,Easytool,DRAFT')
    parser.add_argument('--retrieval_num', type=int, default=5)

    args = parser.parse_args()

    base_path = args.tool_root_dir
    index = build_index(base_path)
    dataset = read_json(f'''dataset/ToolBench/tool_instruction/{args.method}.json''')
    test_data = read_json(f'''dataset/ToolBench/test_data/{args.data_type}.json''')
    progress_file = f'''Toolbnech_dfs_{args.data_type}_{args.model_name}_{args.method}.txt'''

    start_index = get_last_processed_index(progress_file)
    total_files = len(test_data)
    retrieval_num = args.retrieval_num
    ind = start_index
    model_name = args.model_name

    print("-------Start Execution-------")

    task_execution(args.data_type,
            base_path, index, dataset, test_data, progress_file, 
            start_index, total_files, retrieval_num, ind, model_name,args.method)
