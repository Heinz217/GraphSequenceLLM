import torch
import os

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop):
    # TODO:注意看一下各种embedding的格式
    if pretrained_embedding_type == "simteg":
        simteg_sbert = [torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_e5[i]], dim=-1) for i in
                           range(hop + 1)]
        print(f"simteg_sbert: {simteg_sbert}")
        print(simteg_sbert)
        print(f"simteg_e5: {simteg_e5}")
        print(f"pretrained_embs: {pretrained_embs}")
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt")) for i in range(1, hop + 1)]

    return pretrained_embs

def print_every(data_dir, pretrained_embedding_type, hop):
    simteg_sbert_x = torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))
    simteg_sbert_1hop_x = torch.load(os.path.join(data_dir, f"simteg_sbert_1hop_x.pt"))
    simteg_sbert_2hop_x = torch.load(os.path.join(data_dir, f"simteg_sbert_2hop_x.pt"))
    print(f"simteg_sbert_x: {simteg_sbert_x}")
    print(simteg_sbert_x.size())
    print(f"simteg_sbert_1hop_x: {simteg_sbert_1hop_x}")
    print(simteg_sbert_1hop_x.size())
    print(f"simteg_sbert_2hop_x: {simteg_sbert_2hop_x}")
    print(simteg_sbert_2hop_x.size())

    simteg_e5_x = torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))
    simteg_e5_1hop_x = torch.load(os.path.join(data_dir, f"simteg_e5_1hop_x.pt"))
    simteg_e5_2hop_x = torch.load(os.path.join(data_dir, f"simteg_e5_2hop_x.pt"))
    print(f"simteg_e5_x: {simteg_e5_x}")
    print(simteg_e5_x.size())
    print(f"simteg_e5_1hop_x: {simteg_e5_1hop_x}")
    print(simteg_e5_1hop_x.size())
    print(f"simteg_e5_2hop_x: {simteg_e5_2hop_x}")
    print(simteg_e5_2hop_x.size())


if __name__ == '__main__':
    a = load_pretrain_embedding_hop(data_dir='C:/Users/86130/PycharmProjects/pythonProject30/cora-selected', pretrained_embedding_type='simteg', hop=2)
    print(f"pretrained all{a}")
    print_every(data_dir='C:/Users/86130/PycharmProjects/pythonProject30/cora-selected', pretrained_embedding_type='simteg', hop=2)
    print(len(a))
    print(a[0].size())
