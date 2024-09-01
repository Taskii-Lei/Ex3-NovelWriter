import sys
sys.path.append("../")
from novel_writer import Novel_Writer
from src.model_do import load_model
from src.file_rw import try_load_json, write_to_json
import torch
import os

device=torch.device('cuda:0')
device_base = torch.device('cuda:1')

################################## USER SPECIFY #################################
import argparse

parser = argparse.ArgumentParser(description='设置 model型号，levels数量，温度系数')
parser.add_argument('--model_num', type=int, help='模型版本')
parser.add_argument('--level_num', type=int, help='层级深度')
parser.add_argument('--temperature', type=float, help='温度系数')
# parser.add_argument('--v', type=int, help='训练版本', default=62)
# parser.add_argument('--id', type=int, help='生成结果的组别', default=0)
args = parser.parse_args()
parser.set_defaults(level_num=1)

writer_path = f"/your/path/to/finetuned/NovelWriter"
base_path = "/your/path/to/a/generalLLM"
output_dir = f"/path/to/output"


if not os.path.exists(output_dir):
      os.mkdir(output_dir)
### 如果不想指定 premises， 可以直接指定 tag 、 title 和 intro 为 None，writer会自动给出 initialization；
### If you don't want to specify premises, you can directly specify tag, title, and intro as None, and the writer will automatically give initialization;
# _, premises = try_load_json("/path/to/specified/premises") ## for evaluation

novel_tag = "玄幻、惊悚"
novel_title = None
novel_intro = None
################################################################################

writer_model, writer_tokenizer = load_model(model_path=writer_path, device=device)
base_model, base_tokenizer = load_model(model_path=base_path, device=device_base)

# for pre_idx in range(len(premises)):
#     novel_tag = premises[pre_idx]["novel_tag"]
#     novel_title = premises[pre_idx]["novel_title"]
#     novel_intro = premises[pre_idx]["novel_intro"]

output_fold = f"{output_dir}/{novel_title}"
if not os.path.exists(output_fold):
    os.mkdir(output_fold)

config_file = os.path.join(output_fold,'config.json')
config = {
    "fold":output_fold,
    "writer_path": writer_path,
    "novel_type":novel_tag,
    "level_num":args.level_num,
    "novel_intro":novel_intro,
}
write_to_json(config_file, config)

novel_writer = Novel_Writer(writer_model=writer_model, writer_tokenizer=writer_tokenizer, 
                            base_model=base_model, base_tokenzier=base_tokenizer, 
                            novel_tag=novel_tag, novel_title=novel_title, novel_intro=novel_intro, 
                            level_num=args.level_num, output_dir=output_fold, res_tmp=args.temperature, init_tmp=0.95)
novel_writer.write()




    
