import torch
import os


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb



if __name__ == '__main__':
    a = load_pretrain_embedding_graph(data_dir='C:/Users/86130/PycharmProjects/pythonProject30/cora-selected', pretrained_embedding_type='simteg')
    print(f"pretrained all{a}")
    print(a.size())
    # print_every(data_dir='C:/Users/86130/PycharmProjects/pythonProject30/cora-selected', pretrained_embedding_type='simteg', hop=2)
    print(len(a))
    print(a[0].size())
