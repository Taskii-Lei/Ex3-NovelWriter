import os
import time
import sys
sys.path.append("../")
from src.process_specially import remove_ele, preprocess
from src.model_do import load_model, load_embedder, get_api_response
from src.core import content2list, para_length, get_similarity, Seperate_mn, cut_paras, plot_cut
from src.file_rw import try_load_json, write_to_json, write_to_txt
from get_summary import Shrink_Novel
import random

novel_dir = "../RawNovelsDataset"
output_dir = "../ProcessedCorpus"

files = os.listdir(novel_dir)


Novel_ID = []
for i in files:
    id = i.split(".json")[0]
    Novel_ID.append(id)


model_path = "/your/path/to/LLM"
model, tokenizer = load_model(model_path)

embedder_path = '/your/path/to/embedder'
embedder = load_embedder(embedder_path)


if not os.path.exists("./shrinker_info"):
    try:
        os.mkdir("./shrinker_info")
    except:
        print("shrinker info already exists")

random.shuffle(Novel_ID)

for nid in Novel_ID:
    if os.path.exists(f"./shrinker_info/{nid}"):
        print(f"Novel {nid} has already FINISHED!!")
        continue

    write_to_txt("./shrinker_info/going_info.txt", f"it's {nid}\n")
    if not os.path.exists(f"{output_dir}/{nid}"):
        os.mkdir(f"{output_dir}/{nid}")
    if not os.path.exists(f"./shrinker_info/{nid}"):
        os.mkdir(f"./shrinker_info/{nid}")
    json_file = f"{novel_dir}/{nid}.json"
    novel_id, novel = try_load_json(json_file)
    write_to_txt(f"./shrinker_info/output.txt", f"Novel {novel_id} has {len(novel['chapters'])} 章\n", mode="a")
    shrinker = Shrink_Novel(model, tokenizer, embedder, novel_id, novel, output_dir=output_dir)
    info, all_finished = shrinker.resume_chapter_v2()
    write_to_txt(f"./shrinker_info/{nid}/output.txt", info, 'a')
    write_to_txt(f"./shrinker_info/output.txt", info, 'a')
    if all_finished == 0:
        novel_sum = shrinker.shrink_novel()
        print(novel_sum)
    else:
        print(f"Finish: {novel_id} doesn't need to resume!!\n")
    write_to_txt(f"./shrinker_info/output.txt", f"Finish:{novel_id}\n", mode="a")
    write_to_txt("./shrinker_info/going_info.txt", f"---------------{nid} finised!\n")