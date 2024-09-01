import transformers
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import re
from collections import OrderedDict
import os
import random
import pdb
import sys
sys.path.append("../")
from src.file_rw import try_load_json, write_to_json
from src.model_do import load_model, get_api_response, get_writer_response


import re
def get_content_between_a_b(a,b,text):
    result = re.search(f"{a}(.*?){b}", text, re.DOTALL)
    if result == None:
        return None
    else:
        return result.group(1).strip()

def get_content_after_a(a,text):
    result = re.findall(f"(?<={a}).*$", text)
    if len(result) ==0:
        return None
    else:
        return result[0]


def get_person_angle(content):
    content = content.replace("。", "。\n")
    content = content.replace("。\n”", "。”")
    content = content.replace("”", "”\n")
    cont_list = content.split("\n")
    start_me = 0
    for cl in cont_list:
        if "“" in cl or "”" in cl: continue
        if cl.startswith("我"):
            start_me += 1
    if start_me >= 1:
        return "一"
    else:
        return "三"
    

def remove_ele(content, ele, new_ele=""):
    if isinstance(content, str):
        while ele in content:
            content = content.replace(ele, new_ele)
        return content
    if isinstance(content, list):
        while ele in content:
            content.remove(ele)
        return content


def get_good_response(response):
    pattern = r'\d+\.\s'  # 匹配以数字和点号开头的格式，例如 "1. "
    response = re.sub(pattern, '', response)
    pattern = r'[\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u2E3A\u2E3B\u2E41\u2E42\u3014\u3015\u3016\u3017\u3018\u3019\[\]]+|[,.;:?!]+|\n+|-|\（|\）'
    matches = re.split(pattern, response)
    # 去除空字符串和空格
    result = [s.strip() for s in matches if s.strip()]
    return result


class Novel_Writer():
    def __init__(self, writer_model, writer_tokenizer, base_model, base_tokenzier, novel_tag, novel_title=None, novel_intro=None, level_num=1, recall_size=5, person_angle="三", output_dir="./", res_tmp=0.9, init_tmp=0.9):
        self.writer_model = writer_model
        self.writer_tokenzier = writer_tokenizer
        self.base_model = base_model
        self.base_tokenzier = base_tokenzier
        self.novel_tag = novel_tag
        self.novel_title = novel_title
        self.novel_intro = novel_intro
        self.level_num = level_num
        self.recall_size = recall_size      
        self.response_tmp=res_tmp
        self.init_tmp = init_tmp
        self.person_angle = person_angle
        self.output_dir = output_dir
        self.chapter_clk=0
        

    def get_entity_info(self, characters):
        relations = []
        for c in characters:
            if c in self.entity_db:
                c_info = self.entity_db[c]
                relations.append(f"{c} | {c_info}")
            else:
                continue
        if len(relations)==0:
            return '无'
        else:
            return "\n".join(relations)
    

    def novel_init(self,novel_tag,tmp=1):
        prompt= f"""你现在是一个{self.novel_tag}小说作家，要写一个关于{novel_tag}的小说，请给出小说标题和简介"""
        success=False
        index = 0
        while not success:
            text = get_writer_response(model=self.writer_model, tokenizer=self.writer_tokenzier, prompt=prompt,tmp=tmp,top_k=12)
            print(text)
            title = get_content_between_a_b("小说标题：","\n",text)
            summary = get_content_after_a("简介：",text)
            print(title,summary)
            if title is not None and summary is not None:
                success = True
            if not success:
                index+=1
                print("novel init tries",index)
        self.novel_title = title
        self.novel_intro = summary
        return title, summary


    def summary_expand(self,intro):
        prompt= f"""你现在是一名{self.novel_tag}小说作家，正在写一本名为{self.novel_title}的{self.novel_tag}小说，请根据小说简介写出小说大纲，注意故事情节的连贯性：小说简介：\n{intro}"""
        print('='*20+'小说大纲'+'='*20)
        print(prompt)
        success = False
        index  = 0 
        while not success:
            text = get_writer_response(model=self.writer_model, tokenizer=self.writer_tokenzier, prompt=prompt,tmp=self.response_tmp)
            text_list = text.split('\n')
            if len(text_list)>1:
                success = True
            else:
                print(text)
                print('fail')
        content = []
        for item in text_list:
            content.append(item.split("部分：")[1])

        print(text)
        print('='*20)
        return content


    def content_expand(self,content,pre_summary,pre_entity_info, end_flag=False):
        prompt= f"""你现在是一名{self.novel_tag}小说作家，正在写一本名为{self.novel_title}的{self.novel_tag}小说，请联系前文故事情节以及相应的人物或地点的历史信息，根据以下摘要扩展剧情内容：前文故事情节：\n{pre_summary}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n摘要：\n{content}"""
        print('='*20)
        print(prompt)

        success = False
        index  = 0 
        tries = 0
        while not success and tries<10:
            tries+=1
            text = get_writer_response(model=self.writer_model, tokenizer=self.writer_tokenzier, prompt=prompt,tmp=self.response_tmp)
            text_list = text.split('\n')
            if len(text_list)>1:
                success = True
            else:
                print('fail')    
           
        content = []
        for item in text_list:
            content.append(item[5:])
        print('content',prompt,"\n", text)
        print('='*20)
        return content


    def chapter_expand(self,content,pre_summary,pre_entity_info,start_flag=False,end_flag=False):

        print(start_flag,end_flag)
        prompt_start = f"""作为一名{self.novel_tag}小说作家，你正在写一本名为{self.novel_title}的{self.novel_tag}小说开头剧情，请根据开头章节大纲扩写开头剧情：开头章节大纲：\n{content}"""
        prompt = f"""你现在是一名{self.novel_tag}小说作家，正在写一本名为{self.novel_title}的{self.novel_tag}小说，请联系上一章的剧情概要以及相应的人物或地点的历史信息，并根据本章摘要扩写故事情节：上一章的剧情概要：\n{pre_summary}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n本章摘要：\n{content}"""
        prompt_end = f"""作为一名{self.novel_tag}小说作家，你需要为小说{self.novel_title}收尾，以下内容给出了前章大纲、相应人物或地点的历史信息，以及结尾章节大纲，为结尾章节扩写故事情节：前章大纲：\n{pre_summary}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n结尾章节大纲：\n{content}"""

        if start_flag:
            prompt = prompt_start
        elif end_flag:
            prompt = prompt_end
        
       
        success = False
        tries = 0
        while not success and tries<10:
            tries+=1
            text = get_writer_response(model=self.writer_model, tokenizer=self.writer_tokenzier, prompt=prompt,tmp=self.response_tmp)
            text_list = text.split('\n')
            if len(text_list)>=1:
                success = True
            else:
                print('fail')
        print('chapter',prompt,"\n",text)
        return text_list
   

    def para_expand(self,content,pre_summary,pre_entity_info,start_flag=False,end_flag=False):
        print(start_flag,end_flag)
        prompt_start = f"""你正在为你的{self.novel_tag}小说{self.novel_title}写开头段落，请根据开头段剧情大纲，以第{self.person_angle}人称视角扩写开头内容：开头段剧情大纲：\n{content}"""
        prompt= f"""你正在书写{self.novel_tag}小说{self.novel_title}的正文部分，请联系前情提要和人物或地点的历史信息，并根据本段摘要，以第{self.person_angle}人称视角扩写出段落内容：前情提要：\n{pre_summary}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n本段摘要：\n{content}"""
        prompt_end= f"""你的{self.novel_tag}小说{self.novel_title}来到了尾声，以下内容给出了前情提要、人物或地点的历史信息以及结尾段落摘要，请以第{self.person_angle}人称视角为本章写出结尾段落：前情提要：\n{pre_summary}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n本段摘要：\n{content}"""

        if start_flag:
            prompt = prompt_start
        elif end_flag:
            prompt = prompt_end
        success = False
        while not success:
            tries+=1
            text = get_writer_response(model=self.writer_model, tokenizer=self.writer_tokenzier, prompt=prompt,tmp=self.response_tmp)
            if len(text)>100 and get_person_angle(text)==self.person_angle:
                success =True
            else:                
                print('fail')
        print('para',prompt,"\n",text)
        return text


    def chapter_writer(self, chapter_summary, start_flag= False, end_flag=False):
        self.chapters.append(chapter_summary)
        prompt = f"""\n {chapter_summary}\n\n请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：\n人物或地点：请使用顿号连接列出人物或地点名称\n"""
        response = get_api_response(model=self.base_model, tokenizer=self.base_tokenzier, prompt=prompt)
        chapter_characters = get_good_response(response)
        self.recent_visit_level[-1] = chapter_characters + self.recent_visit_level[-1]
        self.recent_visit_level[-1] = list(OrderedDict.fromkeys(self.recent_visit_level[-1]))
        if len(self.recent_visit_level[-1]) >= 100:
            self.recent_visit_level[-1] = self.recent_visit_level[-1][:100]

        chapter_to_visit = list(OrderedDict.fromkeys(chapter_characters + self.recent_visit[:self.recall_size]))
        chapter_relations = self.get_entity_info(chapter_to_visit)
        para_summarys = self.chapter_expand(chapter_summary,self.pre_summary_chapter,chapter_relations,start_flag,end_flag) 
        self.pre_summary_chapter = chapter_summary
        paras = []
        for index,para_summary in enumerate(para_summarys):
            prompt = f""" 
{para_summary}

请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：
人物或地点：请使用顿号连接列出人物或地点名称

"""
            response = get_api_response(model=self.base_model, tokenizer=self.base_tokenzier, prompt=prompt)
            characters = get_good_response(response)
            to_visit = list(OrderedDict.fromkeys(characters + self.recent_visit[:self.recall_size]))
            relations = self.get_entity_info(to_visit)
            is_first = (index==0)
            is_last = (index==(len(para_summarys)-1))

            para = self.para_expand(para_summary,self.pre_summary_para,relations,start_flag = (start_flag and is_first), end_flag=(end_flag and is_last))
            self.pre_summary_para = para_summary
            paras.append(para)

            ## 更新实体数据库 和 最近访问
            prompt = f"""
之前剧情中出现的主要人物、地点，以及各个人物的身份或者与其他人物关系等重要信息如下表所示：
{relations}

当前剧情内容如下所示：
{para}

请提取并简要概括上述当前剧情内容中的主要人物、地点，以及各个人物的身份、与其他人物关系、地点发生事件等重要信息，保留重要身份信息的同时更新上述信息表，并添加新出现的人物、角色或地点的相关信息。请将人物和地点区分开，严格遵从以下格式输出：
人物名｜人物身份、与他人关系等重要信息
地点名｜地点用途、特点等重要信息

"""
            relationship = get_api_response(model=self.base_model, tokenizer=self.base_tokenzier, prompt=prompt)
            relation_list = relationship.split('\n')
            # print(relation_list)

            recent_get_key = []
            for rely in relation_list:
                if not " | " in rely:continue
                rely = remove_ele(rely, " ")
                tmp = rely.split("|")
                tmp = remove_ele(tmp, '')
                tmp = remove_ele(tmp, ' ')
                try:
                    key = tmp[0]
                    value = tmp[1]
                    recent_get_key.insert(0, key)
                    self.entity_db[key] = value
                except:
                    print('para entity fail')

            self.recent_visit = recent_get_key + self.recent_visit
            self.recent_visit = list(OrderedDict.fromkeys(self.recent_visit))
            if len(self.recent_visit) >= 100:
                self.recent_visit = self.recent_visit[:100]
        self.novel = self.novel + [f"\n{'='*50} \n"]
        self.chapter_clk += 1
        self.novel = self.novel + paras


    def level_writer(self, level_summary,level_index,start_flag=False,end_flag=False):

        self.level_summary[level_index].append(level_summary)

        prompt = f""" 
{level_summary}

请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：
人物或地点：请使用顿号连接列出人物或地点名称

"""
        response = get_api_response(model=self.base_model, tokenizer=self.base_tokenzier, prompt=prompt)
        characters = get_good_response(response)
        to_visit = list(OrderedDict.fromkeys(characters + self.recent_visit_level[level_index][:self.recall_size]))
        relations = self.get_entity_info(to_visit)    

        if level_index > 0:
            self.recent_visit_level[level_index-1] = characters + self.recent_visit_level[level_index-1]
            self.recent_visit_level[level_index-1] = list(OrderedDict.fromkeys(self.recent_visit_level[level_index-1]))
            if len(self.recent_visit_level[level_index-1]) >= 100:
                self.recent_visit_level[level_index-1] = self.recent_visit_level[level_index-1][:100]

        if level_index == (self.level_num-1):
            chapters = self.content_expand(level_summary,self.pre_summary_level[level_index],relations)
            for index,chapter_summary in enumerate(chapters):
                is_first = (index==0)
                is_last = (index==(len(chapters)-1))
                print('chapter', index, len(chapters)-1,is_last)
                self.chapter_writer(chapter_summary,is_first and start_flag,is_last and end_flag)
        else:
            contents = self.content_expand(level_summary,self.pre_summary_level[level_index],relations)
            for index,content in enumerate(contents):
                is_first = (index==0)
                is_last = (index==(len(contents)-1))
                print('level',level_index,index,len(contents)-1,is_last)
                self.level_writer(content, level_index+1, is_first and start_flag,is_last and end_flag)           

        self.pre_summary_level[level_index] = level_summary


    def write(self):
        start_id= 0
        self.pre_summary_chapter = '无'
        self.pre_summary_para = '无'
        self.pre_summary_level = ["无" for _ in range(self.level_num+2)]
        self.recent_visit=[]
        self.recent_visit_level=[[] for _ in range(self.level_num+2)]
        self.chapters=[]
        self.entity_db = {}
        self.novel = []
        self.level_summary = [[] for _ in range(self.level_num+2)]

        if self.novel_title is None or self.novel_intro is None:
            title, intro = self.novel_init(self.novel_tag,self.init_tmp)
        summary_file = os.path.join(self.output_dir,'summary.txt')
        fop=open(summary_file,'w')
        fop.write(self.novel_title)
        fop.write(self.novel_intro)


        contents = self.summary_expand(self.novel_intro)
        content_file = os.path.join(self.output_dir,'content.txt')
        fop=open(content_file,'w')
        fop.writelines(contents)
        fop.close()

        for index,content in enumerate(contents):
            is_first = (index==0)
            is_last = (index==(len(contents)-1))
            self.level_writer(content,0,is_first,is_last)
        
        for index, summary in enumerate(self.level_summary):
            level_file = os.path.join(self.output_dir,'level'+str(index)+'.txt')
            fop = open(level_file,'w')
            print(index,len(summary))
            fop.writelines(summary)  
            fop.close()         

        for index,chapter in enumerate(self.chapters):
            chapter_file = os.path.join(self.output_dir,str(index)+'.txt')
            fop = open(chapter_file,'w')
            fop.write(chapter)  
            fop.close()         


        novel_file = os.path.join(self.output_dir,'novel.txt')
        fop = open(novel_file,'w')
        fop.write(self.novel_title+"\n")
        fop.writelines(self.novel)
        write_to_json(f"{self.output_dir}/entity_database.json",self.entity_db)








