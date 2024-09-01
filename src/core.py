import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util

def content2list(content):
    if "\n" in content:
        paralists = content.split("\n")
        while '' in paralists:
            paralists.remove('')
    else:
        paralists = [content]
    return paralists

def para_length(paralists):
    """
    Calculate the cumulative sum of lengths of paragraph strings in the paragraph list and return the array of cumulative lengths.

    Parameters:
    paralists (list): A list of paragrahs for which cumulative lengths need to be calculated.

    Returns:
    numpy.ndarray: An array containing the cumulative sum of lengths of paragraph strings in the input list.
    """
    len_para = []
    len_paras_sum = []
    tmp = 0
    for i in paralists:
        tmp += len(i)
        len_para.append(len(i))
        len_paras_sum.append(tmp)
    len_paras_sum = np.array(len_paras_sum)
    len_mu = np.mean(len_para)
    return len_paras_sum


def get_similarity(paralists, embedder):
    """
    Calculate similarity scores between consecutive pairs of paragraph strings using an embedding model.

    Parameters:
    paralists (list): A list of paragraph strings for which similarity scores need to be calculated.
    embedder: An embedding model used to encode the paragraph strings.

    Returns:
    numpy.ndarray: An array containing the similarity scores between consecutive pairs of paragraph strings, with an additional initial score of 1.
    """
    simi_score = []
    for i in range(len(paralists)-1):
        target_embedding = embedder.encode(paralists[i], convert_to_tensor=True)
        others_embedding = embedder.encode(paralists[i+1], convert_to_tensor=True)
        memory_scores = util.cos_sim(target_embedding, others_embedding)
        numpy_scores = memory_scores.cpu().numpy()
        simi_score.append(numpy_scores)
    simi_score = np.array(simi_score).reshape(1,-1)[0]
    simi_score = np.concatenate(([1], simi_score))
    return simi_score


def max_drop(simi_score, max_drop_threshold=0.1, score_threshold=0.7, mode=1):
    """
    Find the indices where the similarity scores drop below a certain threshold after a significant drop.

    Parameters:
    simi_score (numpy.ndarray): Array of similarity scores between consecutive pairs of strings.
    max_drop_threshold (float): The maximum drop allowed in similarity scores.
    score_threshold (float): The threshold below which the similarity scores are considered low.
    mode (int): The starting index for processing the similarity scores.

    Returns:
    numpy.ndarray: Array containing the indices where the similarity scores drop significantly below the threshold.
    """
    simi_score = simi_score[mode:]
    diff = [0]
    for i in range(len(simi_score)-1):
        diff.append(simi_score[i]-simi_score[i+1])
    diff = np.array(diff)
    idx = (np.argwhere(diff > max_drop_threshold)).flatten()
    del_id = np.where(simi_score[idx] >= score_threshold)
    idx = np.delete(idx, del_id)
    return idx+mode


# def svm_cut(chapter_simi):
#     sub_chapter_simi = np.concatenate((chapter_simi[1:],[0]))
#     data = sub_chapter_simi - chapter_simi
#     chosen_idx = []
#     for i in range(2, len(data)-1):
#         if data[i] >= 0 and data[i-1]<0:
#             chosen_idx.append(i)
#     return chosen_idx


def Seperate_mn(paralists, simi_score, min_length=400, max_length=2000, score_thresh=None):
    """
    Find the cutting points and separate paragraphs based on specified window size and similarity scores.

    Args:
    paralists (list): The list of paragraphs.
    simi_score (list): The list of similarity scores corresponding to paragraphs.
    min_length (int): The minimum length threshold for a paragraph. Default is 400.
    max_length (int): The maximum length threshold for a paragraph. Default is 2000.
    score_thresh (float): The similarity score threshold. Default is None.

    Returns:
    list: A list of selected indices representing the segmented paragraphs.
    """
    len_paras = para_length(paralists)
    if len_paras[-1] <= min_length:
        return [len(paralists)-1]
    min_threshold = 0
    max_threshold = 0
    left = -2
    right = 0
    assert len(len_paras)==len(simi_score), "Dimension Not Match for %d == %d" %(len(len_paras), len(simi_score))
    chosen_idx = []
    while left < right and right <len(paralists) -1:
        right = right if right < len(paralists)-1 else len(paralists)-1
        min_threshold = min(min_threshold + min_length, len_paras[-1]) # type: ignore
        max_threshold = min(min_threshold + max_length, len_paras[-1])
        # print(min_threshold, max_threshold, len_paras - min_threshold)
        try:
            left = np.argwhere((len_paras - min_threshold)>=0)[0][0] # type: ignore
            right = np.argwhere((len_paras - max_threshold)>=0)[0][0] # type: ignore
        except:
            print(f""" 
            len_paras: {len_paras}, min_threshold: {min_threshold}
            len_paras - min_threshold: {len_paras - min_threshold}
            """)
        try: 
            idx = np.argmin(simi_score[left+1:right])+left+1
        except: 
            print(f""" ValueERROR Appears!!  left = {left}, right = {right}""")
            idx = right
            right = right + 1
            continue
        if idx < len(paralists)-3 or len(chosen_idx) < 1:
            chosen_idx.append(idx)
        print(f"left={left}, right={right}, chosen idx={idx}, its score={simi_score[idx]}")

        min_threshold = len_paras[idx-1]
        max_threshold = len_paras[idx-1]
    
    if len(chosen_idx) < 1:
        print(f"WIRED Content!! Pls Check!!{'='*50}\nContentERROR:{paralists}\n{'='*100}")
        chosen_idx.append(len(paralists)-1)
    return chosen_idx



def cut_paras(paralists, cut_idx):
    print(f"Cut idx = {cut_idx}")
    if cut_idx[-1]==0:
        paraparts = [paralists]
        return paraparts
    
    assert len(paralists) >= cut_idx[-1], "List out of RANGE for %d >= %d".format(len(paralists), cut_idx[-1])
    paraparts= []
    for i in range(len(cut_idx)+1):
        if i==0:
            slice=paralists[0:cut_idx[i]]
        elif i==len(cut_idx):
            slice=paralists[cut_idx[i-1]:]
        else:
            slice = paralists[cut_idx[i-1]:cut_idx[i]]
        paraparts.append(slice)
    last = paraparts[-1]
    merge = 0
    if len("\n".join(last)) < 100:
        paraparts[-2] = paraparts[-2]+paraparts[-1]
        paraparts = paraparts[:-1]
        merge = 1
    return paraparts, merge


def plot_cut(simi_score, cut_idx, title:str):
    plt.title(title)
    # print(len(simi_score[cut_idx]), simi_score[cut_idx])
    plt.scatter(range(len(simi_score)), simi_score)
    plt.plot(range(len(simi_score)), simi_score)  
    plt.scatter(cut_idx, simi_score[cut_idx], c="red")
    for k in cut_idx:
        plt.annotate(str(k), (k, simi_score[k]), c="red")
    plt.show()

