import os
import sys
sys.path.append("../")
from src.process_specially import remove_ele, preprocess
from src.model_do import load_model, load_embedder, get_api_response, get_tags
from src.core import content2list, para_length, get_similarity, Seperate_mn, cut_paras, plot_cut
from src.file_rw import try_load_json, write_to_json, write_to_txt
import time

def instructions(novel_type, novel_title, ins:str, person_angle=None):
    if ins=="novel_instruction":
        return f"你现在是一名小说作家，正在写一本名为《{novel_title}》的{novel_type}小说，请根据小说简介写出小说大纲，注意故事情节的连贯性："

    if ins=="in_level_instruction":
        return f"你现在是一名小说作家，正在写一本名为《{novel_title}》的{novel_type}小说，请联系前文故事情节以及相应的人物或地点的历史信息，根据以下摘要扩展剧情内容："

    if ins=="in_chapter_instruction_start":
        return f"作为一名小说作家，你正在写一本名为《{novel_title}》的{novel_type}小说的开头剧情，请根据开头章节大纲扩写开头剧情："
    if ins=="para_group_instruction_start":
        return f"你正在为你的{novel_type}小说《{novel_title}》写开头段落，请根据开头段剧情大纲，以第{person_angle}人称视角扩写开头内容："
    
    if ins=="in_chapter_instruction":
        return f"你现在是一名小说作家，正在写一本名为《{novel_title}》的{novel_type}小说，请联系上一章的剧情概要以及相应的人物或地点的历史信息，并根据本章摘要扩写故事情节："
    if ins=="para_group_instruction":
        return f"你正在书写{novel_type}小说《{novel_title}》的正文部分，请联系前情提要和人物或地点的历史信息，并根据本段摘要，以第{person_angle}人称视角扩写出段落内容："
    
    if ins=="in_chapter_instruction_end":
        return f"作为一名{novel_type}小说作家，你需要为小说《{novel_title}》收尾，以下内容给出了前章大纲、相应人物或地点的历史信息，以及结尾章节大纲，请为结尾章节扩写故事情节："
    if ins=="para_group_instruction_end":
        return f"你的{novel_type}小说《{novel_title}》来到了尾声，以下内容给出了前情提要、人物或地点的历史信息以及结尾段落摘要，请以第{person_angle}人称视角为本章写出结尾段落：" 



def get_person_angle(content):
    '''
    对纯文本收集人称视角。按句切分，句首第一个字是“我”，认定为第一人称文章。
    Collect personal perspectives on plain text. Segment by sentence. If the first word at the beginning of the sentence is "I", it is identified as a first-person article.
    '''
    content = content.replace("。", "。\n")
    content = content.replace("。\n”", "。”")
    content = content.replace("”", "”\n")
    cont_list = content.split("\n")
    start_me = 0
    for cl in cont_list:
        if "“" in cl or "”" in cl: continue
        if cl.startswith("我"):
            start_me += 1
    # percent = start_me / len(cont_list)
    # print("‘我’ 字的频数为：", start_me, percent)
    if start_me >= 1:
        return "一"
    else:
        return "三"



class DC_with_SE():
    def __init__(self, novel_id, input_dir="./", output_dir="./", model=None, tokenizer=None):
        self.novel_id = novel_id
        self.input_dir = f"{input_dir}/{novel_id}"
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer
        nid, novel_level = try_load_json(f"{self.input_dir}/level_{self.novel_id}.json")
        self.novel_title = novel_level['novel_title']
        self.novel_summary = novel_level['summary']

        if "tag" not in novel_level:
            its_tags = get_tags(self.model, self.tokenizer, book_title=self.novel_title, book_summary=self.novel_summary)
            self.novel_tags = "、".join(its_tags)
        else:
            self.novel_tags = "、".join(eval(novel_level["tag"]))
        check_range = 5
        person_3 = 0
        person_1 = 0
        for cr in range(check_range):
            c_id, chapter = try_load_json(f"{self.input_dir}/{cr}.json")
            pg_content = []
            if chapter==None: continue
            try:
                chapter = chapter[0]
            except:
                chapter = chapter
            for pg in chapter['para_groups']:
                pg_content.append(pg['content'])
            if get_person_angle("\n".join(pg_content)) == "一": person_1 += 1
            if get_person_angle("\n".join(pg_content)) == "三": person_3 += 1
        if person_3 > person_1: self.person = "三"
        else: self.person = "一"


    def expand_in_chapter(self, chapter_id, previous, start_flag=False, end_flag=False):
        c_id, chapter = try_load_json(f"{self.input_dir}/{chapter_id}.json")
        if chapter==None: return None, None
        try:
            chapter = chapter[0]
        except:
            chapter = chapter
        para_group_num = chapter['para_groups_num']
        para_groups = chapter['para_groups']
        chapter_summary = chapter['chapter_summary']
        pg_dataset = []
        pg_summaries = []
        for i in range(len(para_groups)):
            pg = para_groups[i]
            entity_info = pg['pre_entity_info']
            content = pg['content']
            summary = pg['summary']
            pg_summaries.append(summary)
            if i-1 >= 0:
                previous = para_groups[i-1]["summary"]
            else:
                previous = previous
            
            if len(entity_info) <= 0:
                entity_info="无"
            
            if i == 0 and start_flag: 
                data =  {
                "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, person_angle=self.person, ins="para_group_instruction_start"),
                "input":f"开头段剧情大纲：\n{summary}",
                "output":content
                }
                pg_dataset.append(data)
            elif end_flag and i==para_group_num-1:
                data =  {
                "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, person_angle=self.person, ins="para_group_instruction_end"),
                "input":f"前情提要：\n{previous}\n\n人物或地点的历史信息：\n{entity_info}\n\n本段摘要：\n{summary}",
                "output":content
                }
                pg_dataset.append(data)
            else:
                data =  {
                    "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, person_angle=self.person, ins="para_group_instruction"),
                    "input":f"前情提要：\n{previous}\n\n人物或地点的历史信息：\n{entity_info}\n\n本段摘要：\n{summary}",
                    "output":content
                }
                pg_dataset.append(data)
        if para_group_num > 1:
            chapter_data = {
                "pre_entity_info":chapter['pre_entity_info'],
                "summary":chapter_summary,
                "content":'\n'.join(pg_summaries)
            }
        else:
            chapter_data=None
        return chapter_data, pg_dataset


    def expand_all_chapters(self, individually_store=False):
        ## 用于将所有章节的正文汇总成一个 json 数据集
        files = os.listdir(self.input_dir)
        All_Chapter_dataset = []
        All_PG_dataset = []
        previous = "无"
        for i in range(len(files)):
            if f"{i}.json" in files:
                if i == 0:
                    chapter_data, pg_dataset = self.expand_in_chapter(chapter_id=i, previous=previous, start_flag=True, end_flag=False)
                    if pg_dataset is not None: All_PG_dataset.extend(pg_dataset)
                    if chapter_data==None: continue
                    chapter_json_data = {
                        "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, ins="in_chapter_instruction_start"),
                        "input":f"开头章节大纲：\n{chapter_data['summary']}",
                        "output":chapter_data['content']
                    }

                    All_Chapter_dataset.append(chapter_json_data)
                    #     All_PG_dataset.extend(pg_dataset)
                elif f"{i+1}.json" in files:
                    chapter_data, pg_dataset = self.expand_in_chapter(chapter_id=i, previous=previous, start_flag=False, end_flag=False)
                    if pg_dataset is not None: All_PG_dataset.extend(pg_dataset)
                    if chapter_data==None: continue
                    if len(chapter_data['pre_entity_info']) <=0 or chapter_data['pre_entity_info']=="":
                        pre_entity_info = "无"
                    else: pre_entity_info = chapter_data['pre_entity_info']

                    chapter_json_data = {
                        "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, ins="in_chapter_instruction"),
                        "input":f"上一章的剧情概要：\n{previous}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n本章摘要：\n{chapter_data['summary']}",
                        "output":chapter_data['content']
                    }
                    All_Chapter_dataset.append(chapter_json_data)
                else:
                    chapter_data, pg_dataset = self.expand_in_chapter(chapter_id=i, previous=previous, start_flag=False, end_flag=True)
                    if pg_dataset is not None: All_PG_dataset.extend(pg_dataset)
                    if chapter_data==None: continue
                    if len(chapter_data['pre_entity_info']) <=0 or chapter_data['pre_entity_info']=="":
                        pre_entity_info = "无"
                    else: pre_entity_info = chapter_data['pre_entity_info']
                    chapter_json_data = {
                        "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title,  ins="in_chapter_instruction_end"),
                        "input":f"前章大纲：\n{previous}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n结尾章节大纲：\n{chapter_data['summary']}",
                        "output":chapter_data['content']
                    }

                    All_Chapter_dataset.append(chapter_json_data)
                    #     All_PG_dataset.extend(pg_dataset)
                # All_PG_dataset.extend(pg_dataset)
                previous = chapter_data["summary"]
                
                # write_to_txt("./test_dc.txt", f"Chapter {i} of Novel {self.novel_id} Done!! \n", mode='a')
            else:
                break
        
        if individually_store:
            write_to_json(f"{self.output_dir}/{self.novel_id}/paragraphs.json", All_PG_dataset)
            write_to_json(f"{self.output_dir}/{self.novel_id}/chapters.json", All_Chapter_dataset)
        return All_Chapter_dataset, All_PG_dataset
    

    def expand_level_and_novel(self):
        files = os.listdir(self.input_dir)
        i = 0
        All_Level_dataset = []
        while f"level_{i}.json" in files:
            _, level = try_load_json(f"{self.input_dir}/level_{i}.json")
            level_dataset = []
            for j in range(len(level)):
                if j-1 >= 0:
                    previous = level[j-1]["content"]
                else:
                    previous = "无"
                source = level[j]['source']
                content = level[j]['content']
                pre_entity_info = "无" if level[j]['pre_entity_info']=="" else level[j]['pre_entity_info']
                for k in range(len(source)):
                    id = source[k]-source[0]
                    content = content.replace(f"第{source[k]+1}部分", f"第{id+1}部分")
                summary = level[j]['summary']
                level_data = {
                    "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, ins="in_level_instruction"),
                    "input":f"前文故事情节：\n{previous}\n\n人物或地点的历史信息：\n{pre_entity_info}\n\n摘要：\n{summary}",
                    "output":content
                }
                level_dataset.append(level_data)
            All_Level_dataset.extend(level_dataset)
            write_to_txt("./test_dc.txt", f"Level {i} of Novel {self.novel_id} Done!! \n", mode='a')
            i+=1
        nid, novel_level = try_load_json(f"{self.input_dir}/level_{self.novel_id}.json")
        content = novel_level['content']
        novel_summary = novel_level['summary']
        novel_data = {
            "instruction":instructions(novel_type=self.novel_tags, novel_title=self.novel_title, ins="novel_instruction"),
            "input":f"小说简介：\n{novel_summary}",
            "output":content
        }
        All_Level_dataset.append(novel_data)

        init_data = {
            "instruction":f"你现在是一个小说作家，要写一个关于{self.novel_tags}的小说，请给出小说标题和简介",
            "input":"",
            "output":f"小说标题：《{self.novel_title}》\n小说简介：{novel_summary}"
        }
        write_to_txt("./test_dc.txt", f"Novel {self.novel_id} Done!! \n", mode='a')
        All_Level_dataset.append(init_data)
        return All_Level_dataset
    

    def expand_b2t(self):
        All_Chapter_dataset, All_PG_dataset = self.expand_all_chapters()
        All_Level_dataset = self.expand_level_and_novel()
        write_to_json(f"{self.output_dir}/{self.novel_id}.json",All_PG_dataset+ All_Chapter_dataset+All_Level_dataset)

