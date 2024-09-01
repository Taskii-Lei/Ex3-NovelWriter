import os
import sys
sys.path.append("../")
import re
import time
from src.model_do import load_model, load_embedder, get_api_response
from src.core import para_length, get_similarity, Seperate_mn, cut_paras, plot_cut
from src.file_rw import try_load_json, write_to_json, write_to_txt
from collections import OrderedDict

def remove_ele(content, ele):
    if isinstance(content, str):
        while ele in content:
            content = content.replace(ele, "")
        return content
    if isinstance(content, list):
        while ele in content:
            content.remove(ele)
        return content


def get_entity_info(entity_db, characters):
    relations = []
    for c in characters:
        if c in entity_db:
            c_info = entity_db[c]
            relations.append(f"{c} | {c_info}")
        else:
            continue
    return "\n".join(relations)


def get_good_response(response):
    pattern = r'\d+\.\s'  # 匹配以数字和点号开头的格式，例如 "1. "
    response = re.sub(pattern, '', response)
    pattern = r'[\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u201C\u201D\u2018\u2019\u3001\u300C\u300D\u300E\u300F\u3008\u3009\u300A\u300B\u300E\u300F\u3010\u3011\u2E3A\u2E3B\u2E41\u2E42\u3014\u3015\u3016\u3017\u3018\u3019\[\]]+|[,.;:?!]+|\n+|-|\（|\）'
    matches = re.split(pattern, response)
    # 去除空字符串和空格
    result = [s.strip() for s in matches if s.strip()]
    return result


def str2dic(strlist):
    lines = strlist.split("\n")
    my_dict = {}
    if strlist=="":return my_dict
    for line in lines:
        parts = line.split(" | ")
        key = parts[0].strip()
        value = parts[1].strip()
        my_dict[key] = value
    return my_dict


class Extract_Entity():
    def __init__(self, model, tokenizer, input_dir, output_dir, novel_id, info_path, resume_from_chapter_id=False):
        self.model = model
        self.tokenizer = tokenizer
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.novel_id = novel_id
        self.entity_db_path = f"{self.output_dir}/{self.novel_id}/entity_database.json"
        self.info_path = info_path
        if resume_from_chapter_id==False:
            init_entity_content = {
            "max_length":0
            }
            write_to_json(self.entity_db_path, init_entity_content)

    def chapters_entity(self, chapter_id, Recent_Visit, reuse_characters=False, recall_size=5):
        info_txt = f"{self.info_path}/{self.novel_id}/entity_info.txt"
        error_txt = f"{self.info_path}/{self.novel_id}/error_info.txt"
        relationship_txt = f"{self.info_path}/{self.novel_id}/relationship.txt"
        _, entity_db = try_load_json(self.entity_db_path)
        if len(Recent_Visit) < recall_size:
            Recent_Visit = Recent_Visit + list(range(recall_size))
            entity_db["Recent_Visit"] = f"{Recent_Visit}"
        Recent_Visit = list(eval(entity_db["Recent_Visit"]))
    
        
        _, chapter_json = try_load_json(f"{self.input_dir}/{self.novel_id}/{chapter_id}.json")
        json_save = f"{self.output_dir}/{self.novel_id}/{chapter_id}.json"
        if chapter_json == None:
            return Recent_Visit
        try:
            chapter = chapter_json[0]
        except:
            chapter = chapter_json
        
        para_groups = chapter['para_groups']
        chapter_summary = chapter['chapter_summary']
        if reuse_characters: chapter_characters = chapter["characters_in_summary"]
        else: 
            prompt = f"""\n{chapter_summary}\n\n请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：\n人物或地点：请使用顿号连接列出人物或地点名称\n\n"""
            response = get_api_response(self.model, self.tokenizer, prompt)
            chapter_characters = get_good_response(response)
        chapter_to_visit = list(OrderedDict.fromkeys(chapter_characters + Recent_Visit[:recall_size]))
        chapter_relations = get_entity_info(entity_db, chapter_to_visit)

        new_pg_groups = []
        for pg in para_groups:
            summary = pg['summary']
            content = pg['content']
            if reuse_characters: characters = pg["characters_in_summary"]
            else:
                prompt = f""" \n{summary}\n\n请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：\n人物或地点：请使用顿号连接列出人物或地点名称\n\n"""
                response = get_api_response(self.model, self.tokenizer, prompt)
                characters = get_good_response(response)
            to_visit = list(OrderedDict.fromkeys(characters + Recent_Visit[:recall_size]))
            relations = get_entity_info(entity_db, to_visit)

            new_pg_data = {
                "id": pg['id'],
                "content": pg['content'],
                "summary": pg['summary'],
                "characters_in_summary":characters,
                "pre_entity_info": relations
            }
            new_pg_groups.append(new_pg_data) 

            ## 更新实体数据库 和 最近访问
            prompt = f"""\n之前剧情中出现的主要人物、地点，以及各个人物的身份或者与其他人物关系等重要信息如下表所示：\n{relations}\n\n当前剧情内容如下所示：\n{content}\n\n请提取并简要概括上述当前剧情内容中的主要人物、地点，以及各个人物的身份、与其他人物关系、地点发生事件等重要信息，保留重要身份信息的同时更新上述信息表，并添加新出现的人物、角色或地点的相关信息。至少要包含{to_visit}。请将人物和地点区分开，严格遵从以下格式输出：\n人物名｜人物身份、与他人关系等重要信息\n地点名｜地点用途、特点等重要信息"""
            relationship = get_api_response(self.model, self.tokenizer, prompt)
            ## ==================================================
            model_reply = f"{'='*100}{self.novel_id} Chapter {chapter_id}\nto_visit：\n{to_visit}\n\nRecent_Visit：\n{Recent_Visit}\n\nRELATIONSHIP：\n原有的关系：{relations}\n\n更新后的：{relationship} \n"
            write_to_txt(relationship_txt, model_reply, 'a')
            ## ==================================================

            relation_list = relationship.split('\n')
            # print(relation_list)
            length = [0]
            recent_get_key = []
            for rely in relation_list:
                if not " | " in rely:continue
                rely = remove_ele(rely, " ")
                if len(rely) > 500: continue
                tmp = rely.split("|")
                tmp = remove_ele(tmp, '')
                tmp = remove_ele(tmp, ' ')
                try:
                    key = tmp[0]
                    value = tmp[1]
                    recent_get_key.insert(0, key)
                    length.append(len(value))
                    entity_db[key] = value
                except:
                    length.append(1)
                    ## ==================================================
                    write_to_txt(error_txt, f"{self.novel_id} Chapter {chapter_id}\n value error:{tmp}\n\n",'a')
                    ## ==================================================
            entity_db['max_length'] = max(entity_db['max_length'],max(length))
            ## ==================================================
            all_items = "\n".join([f"{key} | {value}" for key, value in entity_db.items()])
            info = f"""{'='*100}\n实体数据库：\n {all_items} \n"""
            write_to_txt(info_txt, info, 'a')
            write_to_json(self.entity_db_path, entity_db, 'w')
            ## ==================================================

            Recent_Visit = recent_get_key + Recent_Visit
            Recent_Visit = list(OrderedDict.fromkeys(Recent_Visit))
            if len(Recent_Visit) >= 100:
                Recent_Visit = Recent_Visit[:100]
            entity_db["Recent_Visit"] = f"{Recent_Visit}"

        new_chapter_data = {
            "chapter_title": chapter['chapter_title'],
            "chapter_summary": chapter['chapter_summary'],
            "characters_in_summary":chapter_characters,
            "pre_entity_info":chapter_relations,
            "para_groups_num": chapter['para_groups_num'],
            "para_groups": new_pg_groups
        }
        write_to_json(file_name=json_save, content=new_chapter_data)

        return Recent_Visit


    def level_entity(self, level_id, reuse_characters=False, recall_size=5):
        info_txt = f"{self.info_path}/{self.novel_id}/level_entity_info.txt"
        characters_txt = f"{self.info_path}/{self.novel_id}/level_characters.txt"
        # if len(Level_Recent_Visit) < recall_size:
        #     Level_Recent_Visit = Level_Recent_Visit + list(range(recall_size))
        Level_Recent_Visit = list(range(recall_size))

        _, level_json = try_load_json(f"{self.input_dir}/{self.novel_id}/level_{level_id}.json")
        level_json_save = f"{self.output_dir}/{self.novel_id}/level_{level_id}.json"
        if level_id == self.novel_id:
            summary = level_json['summary']
            if reuse_characters: characters = level_json["characters_in_summary"]
            else: 
                prompt = f""" \n{summary}\n\n请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：\n人物或地点：请使用顿号连接列出人物或地点名称\n\n"""
                response = get_api_response(self.model, self.tokenizer, prompt)
                characters = get_good_response(response)
            novel_data = {
                "novel_title": level_json['novel_title'],
                "source": level_json['source'],
                "pre_entity_info":"无",
                "content": level_json['content'],
                "summary": level_json['summary'],
                "characters_in_summary":characters
            }
            write_to_json(level_json_save, novel_data)
            return 

        new_level_json = []
        for chapter_group in level_json:
            ## source 代表 level_id.json 中每一个 level_group 涵盖了 level_id-1.json 中哪几个部分
            source = chapter_group['source']
            summary = chapter_group['summary']
            if reuse_characters: characters = chapter_group["characters_in_summary"]
            else: 
                prompt = f"""\n{summary}\n\n请提取并简要概括上述内容中出现的主要人物、地点，使用如下格式输出：\n人物或地点：请使用顿号连接列出人物或地点名称\n\n"""
                response = get_api_response(self.model, self.tokenizer, prompt)
                characters = get_good_response(response)
                model_reply = f"{'='*100}\n{'-'*50}RESPONSE：\n{response} \n"
                write_to_txt(characters_txt, model_reply, 'a')
            if chapter_group['id'] == 0:
                ## 第一个 group 是没有实体历史信息的
                new_cg_data = {
                    "id":chapter_group['id'],
                    "source":chapter_group['source'],
                    "pre_entity_info":"无",
                    "content":chapter_group['content'],
                    "summary":chapter_group['summary'],
                    "characters_in_summary":characters
                }
                new_level_json.append(new_cg_data)
                continue

            to_visit = list(OrderedDict.fromkeys(characters + Level_Recent_Visit[:recall_size]))
            ## ==================================================
            write_to_txt(characters_txt, f"Recent_Visit: {Level_Recent_Visit}\n", 'a')
            info = f"""{'='*50}{self.novel_id} Level {level_id}\n人物：{characters} \n"""
            write_to_txt(characters_txt, info, 'a')
            write_to_txt(characters_txt, f"To_Visit: {to_visit}\n", 'a')
            ## ==================================================
            
            ## 这个时候，每个 group 的历史实体信息就不能从 entity database 中得到，只能从上一层中的各个 pre_entity_info 中收集
            ## 上面收集了 当前 group 的 characters 和 recent visit 组成的 to_visit；
            ## 需要在 source 对应的上一层的 group 中收集对应的 to_visit 的历史信息；
            ## info_list 收集初次得到的 to_visit 名单中的信息
            info_list = {}
            recent_get_key = []
            # all_entity = []
            if level_id > 0:
                _, level_pre = try_load_json(f"{self.output_dir}/{self.novel_id}/level_{level_id-1}.json")
            for idx in source:
                if level_id == 0:
                    _, chapter = try_load_json(f"{self.output_dir}/{self.novel_id}/{idx}.json")
                    pre_entity_info = chapter["pre_entity_info"]
                if level_id > 0:
                    pre_entity_info = level_pre[idx]["pre_entity_info"]

                entity_info_dict = str2dic(pre_entity_info)
                # all_entity.extend(pre_entity_info.split("\n"))
                all_keys = ";".join(list(entity_info_dict.keys()))

                for npc in to_visit:
                    npc = str(npc)
                    if npc in all_keys and info_list.get(npc) is None:
                        recent_get_key.insert(0, npc)
                        info_list[npc] = entity_info_dict.get(npc)
                    else:continue
                ## ==================================================
                info = f"""\n\n{'='*50} it's Novel{self.novel_id}, Level_{level_id}, CG {chapter_group['id']}, Source {idx}\n"""
                write_to_txt(info_txt, info)
                write_to_txt(info_txt, '\n'.join([f"{key} | {value}" for key, value in info_list.items()]))
                ## ==================================================
            new_cg_data = {
                "id":chapter_group['id'],
                "source":chapter_group['source'],
                "pre_entity_info":'\n'.join([f"{key} | {value}" for key, value in info_list.items()]),
                "content":chapter_group['content'],
                "summary":chapter_group['summary'],
                "characters_in_summary":characters
            }
            new_level_json.append(new_cg_data)

            
            Level_Recent_Visit = recent_get_key + Level_Recent_Visit
            Level_Recent_Visit = list(OrderedDict.fromkeys(Level_Recent_Visit))
            if len(Level_Recent_Visit) >= 100:
                Level_Recent_Visit = Level_Recent_Visit[:100]
        
        write_to_json(level_json_save, new_level_json)

        return Level_Recent_Visit


    def main_novel_entity(self, resume_start_chapter_id=0, reuse_characters=False):
        all_json_files = os.listdir(f"{self.input_dir}/{self.novel_id}")
        if resume_start_chapter_id == 0:
            Recent_Visit = []
        else:
            _, entity_db = try_load_json(self.entity_db_path)
            Recent_Visit = entity_db["Recent_Visit"]
        cid = resume_start_chapter_id
        while f"{cid}.json" in all_json_files:
            info = f"NOW it's chapter {cid}\n"
            write_to_txt(f"{self.info_path}/{self.novel_id}/going_info.txt", info, 'a')
            Recent_Visit = self.chapters_entity(chapter_id=cid, Recent_Visit=Recent_Visit, reuse_characters=reuse_characters)
            cid += 1

        lid = 0
        while f"level_{lid}.json" in all_json_files:
            info = f"NOW it's level_{lid}\n"
            write_to_txt(f"{self.info_path}/{self.novel_id}/going_info.txt", info, 'a')
            self.level_entity(level_id=lid, reuse_characters=reuse_characters)
            lid += 1
        self.level_entity(level_id=self.novel_id, reuse_characters=reuse_characters)
        info = f"Novel {self.novel_id} Done!!\n"
        write_to_txt(f"{self.info_path}/going_info.txt", info, 'a')
    

    def main_chapters_entity(self, resume_start_chapter_id=0, recover_already_have=False):
        all_json_files = os.listdir(f"{self.input_dir}/{self.novel_id}")
        if resume_start_chapter_id == 0:
            Recent_Visit = []
        else:
            _, entity_db = try_load_json(self.entity_db_path)
            Recent_Visit = entity_db["Recent_Visit"]
        cid = resume_start_chapter_id
        while f"{cid}.json" in all_json_files:
            info = f"NOW it's chapter {cid}\n"
            write_to_txt(f"{self.info_path}/{self.novel_id}/going_info.txt", info, 'a')
            Recent_Visit = self.chapters_entity(chapter_id=cid, Recent_Visit=Recent_Visit, resume=recover_already_have)
            cid += 1

        info = f"Novel {self.novel_id} chapters Done!!\n"
        write_to_txt(f"{self.info_path}/going_info.txt", info, 'a')

    
    def main_level_entity(self, resume_start_level_id=0, recover_already_have=False):
        all_json_files = os.listdir(f"{self.input_dir}/{self.novel_id}")
        lid = resume_start_level_id
        while f"level_{lid}.json" in all_json_files:
            info = f"NOW it's level_{lid}\n"
            write_to_txt(f"{self.info_path}/{self.novel_id}/going_info.txt", info, 'a')
            self.level_entity(level_id=lid, resume=recover_already_have)
            lid += 1
        self.level_entity(level_id=self.novel_id, resume=recover_already_have)
        info = f"Novel {self.novel_id} Level Done!!\n"
        write_to_txt(f"{self.info_path}/going_info.txt", info, 'a')
    