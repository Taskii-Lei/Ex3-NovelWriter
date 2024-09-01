import os
import time
import sys
sys.path.append("../")
from src.model_do import load_model, get_api_response
from src.process_specially import remove_ele, preprocess
from src.file_rw import try_load_json, write_to_json, write_to_txt
from Entity_info import Extract_Entity
from collections import OrderedDict
from tag_generation import main_tag

input_dir = "../ProcessedCorpus" # "/your/path/to/your/corpus"
output_dir = "../ProcessedCorpus" # "/your/path/to/save/corpus" 


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_path = "/your/path/to/LLM"
model, tokenizer = load_model(model_path)

all_novel = os.listdir(input_dir)

ing_info_path = f"./your/path/to/save/processing/info" ## recommend "./shrinker_info/novel_title"
if not os.path.exists(ing_info_path):
    os.mkdir(ing_info_path)

import time
import random
random.shuffle(all_novel)
for novel_id in all_novel:
    novel_info_path = f"{ing_info_path}/{novel_id}"
    novel_output_path = f"{output_dir}/{novel_id}"

    if not os.path.exists(novel_info_path):
        os.mkdir(novel_info_path)
    if not os.path.exists(f"{output_dir}/{novel_id}"):
        os.mkdir(f"{output_dir}/{novel_id}")
    entity_extractor = Extract_Entity(model, tokenizer, input_dir, output_dir, novel_id, info_path = ing_info_path, resume_from_chapter_id=False)
    entity_extractor.main_novel_entity()
    main_tag(model, tokenizer, f"{output_dir}/{novel_id}", novel_id)