import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
from text2vec import SentenceModel

def load_model(model_path, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    return model, tokenizer


def load_embedder(embedder_path):
    try:
        embedder = SentenceTransformer(embedder_path)
    except:
        embedder = SentenceModel(embedder_path)
    return embedder


def get_api_response(model, tokenizer, prompt):
    # try:
    messages = []
    messages.append({"role": "user", "content":prompt})
    response = model.chat(tokenizer, messages)
    # except:
    #     response, _ = model.chat(tokenizer, prompt, history=None)
    return response

def get_writer_response(model, tokenizer,  prompt, tmp=0.9, top_k=5, max_new_tokens=4096):
    messages = []
    messages.append({"role": "user", "content": prompt})
    input_ids = model._build_chat_input(tokenizer, messages, max_new_tokens)
    generation_config = model.generation_config
    generation_config.temperature = tmp
    generation_config.top_k = top_k
    outputs = model.generate(input_ids,generation_config)
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

    
def get_tags(model, tokenizer, book_title, book_summary):
    """
    Just in case a novel doesn't have theme tags.
    """
    tags = "战争, 武侠, 仙侠, 历史, 科幻, 奇幻, 惊悚, 侦探, 恐怖, 浪漫, 悬疑, 言情, 都市, 青春, 社会, 传记, 自传, 心理, 玄幻, 修真, 游戏, 体育, 军事"
    prompt = f"""
小说《{book_title}》的简介如下：
{book_summary}

请根据上述小说标题以及简介，从 {tags} 中挑选三个符合该小说的标签。
"""
    its_tag = []
    response = get_api_response(model, tokenizer, prompt)
    for i in tags.split(", "):
        if i in response:
            its_tag.append(i)
    return its_tag