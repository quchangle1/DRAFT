import random
import json
import re
from itertools import combinations
import sys
import openai
import os
import time
import importlib
import requests
import numpy as np
from pydantic import BaseModel
from typing import Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


rapidapi_key = " "
openai.api_base = " "
openai.api_key = " "

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str

def openai_response(messages, temperature, top_p, max_tokens, model): 
    try:
        ans = get_response(messages, temperature, top_p, max_tokens, model)
        cleaned_text = ans.strip("`json\n").strip("`\n")
        ans = json.loads(cleaned_text)
        return ans
    except Exception as e: 
        print(f"Caught an exception of type: {type(e)}") 
        print("pausing")

def openai_embedding(text):
    try:
        ans = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return ans['data'][0]['embedding']
    except Exception as e: 
        print(f"Caught an exception of type: {type(e)}") 
        print("pausing")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

def compute_similarity_and_bleu(reference_sentence, candidate_sentence):
    reference_sentence_embedding = openai_embedding(reference_sentence)
    candidate_sentence_embedding = openai_embedding(candidate_sentence)
    similarity = cosine_similarity(reference_sentence_embedding, candidate_sentence_embedding)

    reference = [reference_sentence.lower().split()]
    candidate = candidate_sentence.lower().split()

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    delta = (bleu_score + similarity) / 2
    return delta


with open('prompts/Explorer.txt', 'r') as file:
    example_prompt_template = file.read()
    example_prompt, example_prompt_follow = example_prompt_template.split('=========')

with open('prompts/Analyzer.txt', 'r') as file:
    suggestion_prompt_template = file.read()
    suggestion_prompt, suggestion_prompt_follow = suggestion_prompt_template.split('=========')

with open('prompts/Rewriter.txt', 'r') as file:
    rewrite_prompt_template = file.read()
    rewrite_prompt, rewrite_prompt_follow = rewrite_prompt_template.split('=========')

with open('dataset/ToolBench/tool_instruction/Initial.json', 'r', encoding='utf-8') as file:
    tools = json.load(file)

temperature = 0.2
top_p = 1
max_tokens = 2000
model = "gpt-4o-2024-08-06"
episodes=5

for tool in tools.values():
    tool_save =tool
    tool_category=tool['category']
    tool_name=tool['tool_name']
    for name, api_info in tool["tool_guidelines"].items():
        api_name=api_info['name']
        last_tool_description=api_info['description']
        required_parameters=api_info['required_parameters']
        optional_parameters=api_info['optional_parameters']
        explored_queries=[]
        explored_queries_embeddings=[]
        explored_examples=[]
        suggestions=[]
        rewrite_description_history=[]
        rewrite_description_history.append(last_tool_description)
        rewrite_agent_history=[]
        suggestion_from_rewrite_agent=''
        for episode in range(episodes):
            tool_info = {'category': tool_category, 'name': api_name, 'description': rewrite_description_history[-1], 'required_parameters': required_parameters, 'optional_parameters': optional_parameters}
            tool_description = str(tool_info)
            explore_prompt = example_prompt.replace('{Tool Description}', tool_description)
            if len(explored_queries) > 0:
                explore_prompt_follow = example_prompt_follow.replace('{Explored queries}', str(explored_queries))
                explore_prompt_follow = explore_prompt_follow.replace('{Suggestions}', suggestion_from_rewrite_agent)
                explore_prompt = explore_prompt + explore_prompt_follow
                for t in range(3):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": explore_prompt}
                    ]
                    example_ans = openai_response(messages, temperature, top_p, max_tokens, model)
                    cur_embedding = openai_embedding(example_ans['User Query'])
                    similarity = [cosine_similarity(emb, cur_embedding) for emb in explored_queries_embeddings]
                    if all(sim < 0.9 for sim in similarity): 
                        break
                    else:
                        explore_prompt += f"\nYour last generate query '{example_ans['User Query']}' is too similar to the previous ones. Please generate a different query."
            else :
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": explore_prompt}
                ]
                example_ans = openai_response(messages, temperature, top_p, max_tokens, model)
            explored_queries.append(example_ans['User Query'])
            explored_queries_embeddings.append(openai_embedding(example_ans['User Query']))
            
            cate = tool_category
            tool_name = change_name(standardize(tool_name))
            api_name = change_name(standardize(api_name))
            parameters = example_ans['Parameters']
            payload = {
                "category": cate,
                "tool_name": tool_name,
                "api_name": api_name,
                "tool_input": parameters,
                "strip": "filter",
                "rapidapi_key": rapidapi_key,
            }

            response = get_rapidapi_response(payload)
            example_ans['API_Response'] = response
            explored_examples.append(example_ans)

            suggestion_prompt_temp = suggestion_prompt.replace('{Tool Description}', tool_description)
            suggestion_prompt_temp = suggestion_prompt_temp.replace('{usage_example}', str(example_ans))
            if len(rewrite_description_history) > 1:
                suggestion_prompt_follow_temp = suggestion_prompt_follow.replace('{History}', str(rewrite_description_history))
                suggestion_prompt_temp += suggestion_prompt_follow_temp
            messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": suggestion_prompt_temp}
                    ]
            suggestion_ans = openai_response(messages, temperature, top_p, max_tokens, model)
            suggestions.append(suggestion_ans)
            

            rewrite_prompt_temp = rewrite_prompt.replace('{Tool Description}', tool_description)
            rewrite_prompt_temp = rewrite_prompt_temp.replace('{usage_example}', str(example_ans))
            rewrite_prompt_temp = rewrite_prompt_temp.replace('{Suggestions}', suggestion_ans['Suggestions for tool description'])
            rewrite_prompt_temp = rewrite_prompt_temp.replace('{tool_description}', tool_info['description'])
            if len(rewrite_description_history) > 1:
                rewrite_prompt_follow_temp = rewrite_prompt_follow.replace('{History}', str(rewrite_description_history))
                rewrite_prompt_temp += rewrite_prompt_follow_temp
            messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rewrite_prompt_temp}
                    ]
            rewrtite_ans = openai_response(messages, temperature, top_p, max_tokens, model)
            rewrite_description_history.append(rewrtite_ans['Rewritten description'])
            last_tool_description = rewrtite_ans['Rewritten description']
            suggestion_from_rewrite_agent=str(rewrtite_ans['Suggestions for exploring'])
            rewrite_tool={}
            rewrite_tool['tool_description'] = rewrtite_ans
            rewrite_agent_history.append(rewrite_tool)

            if len(rewrite_description_history) > 1:
                reference_sentence = rewrite_description_history[-2]
                candidate_sentence = rewrite_description_history[-1]
                delta = compute_similarity_and_bleu(reference_sentence, candidate_sentence)
                if delta < 0.75:
                    break  

        api_info['description'] = rewrite_description_history[-1]
    
    tool_doc = str(tool)

    with open('prompts/rewrite_tool_doc.txt', 'r') as file:
        prompt_template = file.read()
    rewrite_tool_prompt = prompt_template.replace('{Tool Description}', tool_doc)
    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rewrite_tool_prompt}
                    ]
    tool['tool_description'] = openai_response(messages, temperature, top_p, max_tokens, model)['tool_description']

    with open("DRAFT.json", 'a', encoding='utf-8') as output_data_file:
        json.dump(tool, output_data_file, ensure_ascii=False, indent=4)
        output_data_file.write(',')
        output_data_file.write('\n')
