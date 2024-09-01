from Dataset_Constructor import DC_with_SE
from src.model_do import load_model, get_api_response
import os
from src.file_rw import try_load_json, write_to_json, write_to_txt


model_path = "/your/path/to/your/general/LLM"
model, tokenizer = load_model(model_path)

input_dir = f"../ProcessedCorpus/"
output_dir = f"../FinalCorpus/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
novel_files = os.listdir(input_dir)
Done_files = os.listdir(output_dir)

import random
random.shuffle(novel_files)
if not os.path.exists("./entity_expand_start_end"):
    os.mkdir("./entity_expand_start_end")

for nid in novel_files:
    if nid in Done_files:
        continue
    # try:
    DC = DC_with_SE(nid, input_dir, output_dir, model=model, tokenizer=tokenizer)
    DC.expand_b2t()
    info = f"Novel {nid} has Done!!\n"
    write_to_txt("./entity_expand_start_end/main_expand_SE.txt", info, 'a')
    Done_files.append(nid)
    # except:
    #     print(nid, " raises error!")
    #     Done_files.append(nid)