import json

def try_load_json(json_file):
    novel_id = json_file.split("/")[-1][:-5]
    try:
        with open(json_file,"r") as f:
            novel = json.load(f)
            return novel_id, novel
    except json.decoder.JSONDecodeError:
        # print(json_file," gets ERROR")
        return None, None



def write_to_json(file_name, content, mode='w', indent=4, write_signal=True):
    if write_signal:
        with open(file_name, mode, encoding='utf-8') as f:
            json.dump(content,f, ensure_ascii=False, indent=indent)
    else: return      



def write_to_txt(file_name, content, mode='a', write_signal=True):
    if write_signal:
        with open(file_name, mode, encoding='utf-8') as f:
            f.write(f"{content}")
    else: return 

