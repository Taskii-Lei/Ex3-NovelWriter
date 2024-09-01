import os
import sys
sys.path.append("../")
from src.process_specially import remove_ele, preprocess, de_preprocess
from src.model_do import load_model, load_embedder, get_api_response, get_tags
from src.core import content2list, para_length, get_similarity, Seperate_mn, cut_paras, plot_cut
from src.file_rw import try_load_json, write_to_json, write_to_txt
import time


'''
Distinguish some nouns
A novel has multiple chapters, each chapter has many paragraphs
In a certain chapter, make a paragraph-level truncation, multiple related paragraphs form para_group, and make an abstract to get para_group_sum
para_group_sum no longer continue to divide, directly combine to make a summary to get the summary of this chapter chapter_sum
After getting the summary of all chapters, truncate the chapter summary, and form a group of multiple similar chapter summaries to get chapter_groups, and get the summary chapter_group_sum
Then summarize all the chapter_group_sum to get a summary of this novel novel_summary
'''

instructions={
        "para_group_instruction":"请你给以下段落剧情写一段简单的摘要：",
        "in_chapter_instruction":"请你给以下章节剧情写一段简单的摘要，尽量涵盖整体剧情：",
        "in_level_instruction":"请用一段话概括以下内容的主要剧情，尽量涵盖整体剧情：",
        "novel_instruction":"请为以下内容中的主要故事情节写一段摘要，尽量涵盖整体剧情："
}

class Shrink_Novel:
    def __init__(self, model, tokenizer, embedder, novel_id, novel, output_dir="./"):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.novel_id = novel_id
        self.novel = novel
        self.output_dir = f"{output_dir}/{novel_id}"

    def shrink_para_groups(self, 
        content:str,
        cut_way='mn', 
        min_length_threshold=400,
        max_length_threshold=1800,
        min_times = 4,
        max_times = 10,
        will_plot=True,
        verbose=False,
        title=None, 
        instruction=instructions["para_group_instruction"],
        resume_start = 0,
        level_shrinker=False
        ):
        if isinstance(content, str):
            tmp_content = preprocess(content)
            if "\n" in tmp_content:
                content = tmp_content
            else:
                content = "”\n“".join(content.split("”“"))
                content = "。\n".join(content.split("。"))
                content = "。”\n".join(content.split("。\n”"))
            paralists = content2list(content)
            conlen = len(content)
        elif isinstance(content, list):
            paracontent = "\n".join(content)
            return self.shrink_para_groups(content=paracontent, cut_way=cut_way, instruction=instruction)
        else:
            write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", f"Novel {self.novel_id}  ERROR!Check the format of your CONTENT! \n", mode='a')
            raise TypeError
        
        average_length = conlen/len(paralists)
        try:
            if average_length >= 50:
                min_length = min(int(min_times*average_length), min_length_threshold)
                max_length = min(int(max_times*average_length), max_length_threshold)
            else:
                min_length = min_length_threshold
                max_length = max_length_threshold
        except:
            min_length = 0
            max_length = conlen
        simi_score = get_similarity(paralists, self.embedder)
        chosen_idx = Seperate_mn(paralists, simi_score, min_length=min_length, max_length=max_length)
        para_groups, merge = cut_paras(paralists, chosen_idx)
        if level_shrinker: merge=0
        if merge == 0: cut_idx = [0]+chosen_idx+[len(paralists)]
        if merge == 1: cut_idx = [0]+chosen_idx[:-1]+[len(paralists)]

        para_groups_sum = []
        for i in range(len(para_groups)):
            parapart_content = "\n".join(para_groups[i][:])
            prompt = f"""{instruction}\n{parapart_content}\n\n"""
            response = get_api_response(self.model, self.tokenizer, prompt)
            response = remove_ele(response, "\n")
            para_groups_sum.append(response)
            if verbose:
                print("="* 50, " PROMPT ", "="* 50, "\n", prompt, "="*100)
                print(len(res), res)
                print("="*100)
        if will_plot: 
            plot_cut(simi_score, chosen_idx, title=title)
    
        return para_groups, para_groups_sum, cut_idx


    def shrink_chapter(self, chapter, chapter_id, min_length_threshold = 400, max_length_threshold = 1800, wt_chapter_json=True):
        ## To obtain the summary of a chapter
        if self.novel != None:
            novel_title = self.novel["book_title"]
        chapter_title = chapter['chapter_title']
        chapter_content = chapter['chapter_content']
        if isinstance(chapter, str):
            chapter_content = chapter
            novel_title = "Unknown"
            chapter_title = "Unkonwn"

        para_groups, para_groups_sum, _ = self.shrink_para_groups(
            content=chapter_content, 
            cut_way="mn", 
            min_length_threshold = min_length_threshold, ## 最小文本长度不超过此阈值
            max_length_threshold = max_length_threshold, ## 最大文本长度不超过此阈值
            verbose=False,
            title=f"Similarity of Chapter {chapter_id}",
        )
        print(para_groups_sum)
        all_paras_sum = " ".join(para_groups_sum)
        prompt = f"""{instructions["in_chapter_instruction"]}\n{all_paras_sum}\n\n"""
        response = get_api_response(self.model, self.tokenizer, prompt)
        chapter_sum = remove_ele(response, "\n")

        if wt_chapter_json:
            if not os.path.exists(self.output_dir): # type: ignore
                os.mkdir(self.output_dir) # type: ignore
            
            chapter_json_file = f"{self.output_dir}/{chapter_id}.json"
            
            para_group_json = []
            for i in range(len(para_groups_sum)):
                data = {
                    "id":i,
                    "content": de_preprocess("\n".join(para_groups[i])),
                    "summary": para_groups_sum[i]
                }
                para_group_json.append(data)
            chapter_json_data = [
                {
                    "chapter_title":chapter_title, 
                    "chapter_summary":chapter_sum, 
                    "para_groups_num": len(para_groups_sum),
                    "para_groups": para_group_json
                }
            ]
            write_to_json(file_name=chapter_json_file,
                          content=chapter_json_data,
                          write_signal=wt_chapter_json)
        return chapter_sum
    

    def resume_chapter_v2(self):
        already_have = os.listdir(f"{self.output_dir}")
        all_chapters = self.novel["chapters"]
        missing_chapters = []
        all_finished = 0
        for idx in range(len(all_chapters)):
            if f"{idx}.json" not in already_have:
                missing_chapters.append(idx)

        if len(already_have) >= 1:
            if len(missing_chapters) > 0:
                info = f"Resume chapter {missing_chapters}!\n"
                write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", info, 'a')
            else:
                info = f"Chapters of Novel {self.novel_id} has done! Levels left!\n"
                if f"level_{self.novel_id}.json" in already_have:
                    all_finished = 1
                return info, all_finished
        else:
            info = f"Chapters of Novel {self.novel_id} hasn't started yet!!\n"
            write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", info, 'a')


        for idx in missing_chapters:
            sub_info = f"chapter {idx} of Novel {self.novel_id} starts summarizing !! \n"
            write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", sub_info, 'a')
            chapter = all_chapters[idx]
            chapter_summary = self.shrink_chapter(chapter, idx, wt_chapter_json=True)
            sub_info = "Finished\n"
            write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", sub_info, 'a')
        info = f"Chapters of Novel {self.novel_id} has done!\n"
        return info, all_finished


    def shrink_novel(self, end_length = 1600, wt_json_file=True):
        novel_title = self.novel["book_title"]
        all_chapters = self.novel["chapters"]
        all_part_summary = []
        for idx in range(len(all_chapters)):
            # chapter = all_chapters[idx]
            # chapter_summary = self.shrink_chapter(chapter, idx, wt_chapter_json=wt_json_file)
            _, chapter = try_load_json(f"{self.output_dir}/{idx}.json")
            if chapter==None:
                continue
            try:chapter=chapter[0]
            except:chapter=chapter
            chapter_summary = chapter["chapter_summary"]
            all_part_summary.append(f"{chapter_summary}")
        
        all_parts_sum_content = "\n".join(all_part_summary)
        level = 0
        while(len(all_parts_sum_content) > end_length):
            chapter_groups, chapter_groups_sum, cut_idx = self.shrink_para_groups(
                content=all_parts_sum_content,
                min_length_threshold=400,
                max_length_threshold=end_length,
                min_times = 3,
                max_times = 8,
                instruction=instructions["in_level_instruction"],
                level_shrinker=True
            )
            if wt_json_file:
                file_name = f"{self.output_dir}/level_{level}.json"
                level_data = []
                for k in range(len(chapter_groups_sum)):
                    for cgs in range(len(chapter_groups[k])):
                        # print(cut_idx[k]+cgs+1, chapter_groups[k][cgs])
                        chapter_groups[k][cgs] = f"第{cut_idx[k]+cgs+1}部分：" + chapter_groups[k][cgs]
                    data = {
                        "id":k,
                        "source":list(range(cut_idx[k],cut_idx[k+1])),
                        "content":"\n".join(chapter_groups[k]),
                        "summary":chapter_groups_sum[k]
                    }
                    level_data.append(data)
                write_to_json(file_name, level_data)
            
            all_parts_sum_content = "\n".join(chapter_groups_sum)
            sub_info = f"level {level} of Novel {self.novel_id} Finishs summarizing !! \n"
            write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", sub_info, 'a')
            level+=1
            
        
        prompt = f"""{instructions["novel_instruction"]}\n{all_parts_sum_content}\n\n"""
        response = get_api_response(self.model, self.tokenizer, prompt)
        response = remove_ele(response, "\n")
        its_tags = get_tags(self.model, self.tokenizer, book_title=novel_title, book_summary=response)

        apsc_list = content2list(all_parts_sum_content)
        for k in range(len(apsc_list)):
            apsc_list[k] = f"第{k+1}部分：" + apsc_list[k] # type: ignore
        novel_data = {
            "novel_title":novel_title,
            "tag": f"{its_tags}",
            "source":list(range(len(apsc_list))),
            "content":"\n".join(apsc_list),
            "summary":response
        }
        novel_file = f"{self.output_dir}/level_{self.novel_id}.json"
        write_to_txt(f"./shrinker_info/{self.novel_id}/output.txt", f"Novel {self.novel_id} has {level} levels in total! Finished!!\n", mode="a")
        write_to_json(novel_file, novel_data)
            



