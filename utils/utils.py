import torch
from constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN


def get_model_name_from_path(model_path):
    """
    进行评估时会用到这个
    :param model_path:
    :return:
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def tokenizer_graph_token(prompt, tokenizer, graph_token_index=GRAPH_TOKEN_INDEX, return_tensors=None):
    """
    同样在评估时会用到这个
    :param prompt:这里的输入应当是一个列表，列表中的元素为token_id，graph则被定义为<graph>
    :param tokenizer:
    :param graph_token_index:
    :param return_tensors:
    :return:
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)]
    # 这里是将prompt按照DEFAULT_GRAPH_TOKEN进行切分
    # DEFAULT_GRAPH_TOKEN = "<graph>"

    # 内部函数：在每个块之间插入分隔符
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]  # 这里是将prompt的每个部分之间插入分隔符

    input_ids = []
    offset = 0
    # 如果第一个块的第一个token是BOS（句子开始）token，则将其添加到input_ids中
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 将分隔符插入到每个块之间，并将结果添加到input_ids中
    for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    # 实际上相当于将prompt按照graph_token_index进行切分，分成多个块，按照块的顺序将每个块的token添加到input_ids中

    # 如果指定了返回张量类型，则将input_ids转换为相应的张量
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)