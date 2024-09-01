import os
import sys
sys.path.append("../")
from src.model_do import load_model, get_api_response
from src.file_rw import try_load_json, write_to_json

"""
以防有些纯文本小说没有tags，需要生成一下；
In case some raw text novels do not have tags, they need to be generated.
"""
def tag_generation(model, tokenizer, book_title, novel_summary):
    tags = "战争, 武侠, 历史, 科幻, 奇幻, 惊悚, 侦探, 恐怖, 浪漫, 言情, 都市, 青春, 社会, 传记, 自传, 心理, 玄幻, 修真"
    prompt = f"""小说《{book_title}》的简介是：\n{novel_summary}\n\n请从 {tags} 中挑选三个符合该小说的tag."""
    its_tag = []
    response = get_api_response(model, tokenizer, prompt)
    for i in tags.split(", "):
        if i in response:
            its_tag.append(i)
    return its_tag


def main_tag(model, tokenizer, path, novel_id):
    _, novel = try_load_json(f"{path}/level_{novel_id}.json")
    its_tag = tag_generation(model, tokenizer, book_title= novel["novel_title"], novel_summary=novel['summary'])
    novel["tag"] = f"{its_tag}"
    write_to_json(f"{path}/level_{novel_id}.json", novel)
    print(f"{path}/level_{novel_id} generated tags!" )


