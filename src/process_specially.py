import os
import re


"""
Processing online novel data is very cumbersome. Here are some special processing functions that may be used
"""

def remove_puncs(response):
    pattern = r'\d+\.\s'  # Matches formats that begin with a number and a period, such as "1."
    response = re.sub(pattern, '', response)
    pattern = r'[\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u2E3A\u2E3B\u2E41\u2E42\u3014\u3015\u3016\u3017\u3018\u3019\[\]]+|[,.;:?!]+|\n+|-|\（|\）'
    matches = re.split(pattern, response)
    # Remove empty strings and spaces
    result = [s.strip() for s in matches if s.strip()]
    return result


def remove_ele(content, ele, new_ele=""):
    if isinstance(content, str):
        while ele in content:
            content = content.replace(ele, new_ele)
    if isinstance(content, list):
        for i in range(len(content)):
            while ele in content[i]:
                content[i] = content[i].replace(ele,new_ele)
    return content


import random

def preprocess(content):
    random_num = random.random()
    assert isinstance(content, str), "Input is NOT STRING"
    if "：\n" in content: 
        content = content.replace("：\n", "：")
    if "”\n“" in content:
        if random_num>0.5:
            content = content.replace("”\n“", "” “")
    return content


def de_preprocess(content):
    assert isinstance(content, str), "Input is NOT STRING"
    while "” “" in content:
        content = content.replace("” “", "”\n“")
    return content
