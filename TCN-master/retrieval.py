import torch
import numpy as np
from TCN.word_cnn.model import TCN
import datautils
# from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from tqdm import tqdm
import torch
import numpy as np
import argparse
import yaml
from TCN.word_cnn.model import TCN
import datautils

def all_retrieval(model, config):
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    num = config["retrieval"]["num"]
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"],data_path="electricity_2012_hour.csv", flag='train',size=[L, 0, L])
    all_repr=torch.load('../data/TCN/ele_hisvec_list.pt')
    references=[]
    with torch.no_grad():
        for i in tqdm(range(len(train_set) - L - H + 1)):
            x=train_set.data_x[i:i+L]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            # x_vec = model.encode(x)
            x_vec = model(x)
            k=x_vec.shape[-2]
            l=x_vec.shape[-1]
            x_vec=x_vec.reshape(1,k*l)
            # all_repr = all_repr.reshape(config["retrieval"]["length"],k*l)
            all_repr = all_repr.reshape(-1,k*l)
            distances=torch.norm(x_vec.cpu() - all_repr,dim=1)
            _, idx=torch.topk(-1*distances, num)
            references.append(idx.int())
        references = torch.cat(references, dim=0)
        torch.save(references, config["path"]["ref_path"])
    return references

def all_encode(model,config):
    hisvec_list=[]
    reference_list=[]
    L=config["retrieval"]["L"]
    H=config["retrieval"]["H"]
    train_set = datautils.Dataset_Electricity(root_path=config["path"]["dataset_path"], data_path="electricity_2012_hour.csv",flag='train',size=[L, 0, L])
    with torch.no_grad():
        for i in tqdm(range(len(train_set) - L - H + 1)):
        # for i in range(len(train_set)):
            x=train_set.data_x[i:i+L]
            y=train_set.data_x[i+L:i+L+H]
            x=x[np.newaxis, :, :]
            x=torch.tensor(x).transpose(1, 2).to(config["retrieval"]["device"])
            # x_vec = model.encode(x)
            x_vec = model(x)
            hisvec_list.append(x_vec.cpu())
    hisvec_list = torch.cat(hisvec_list, dim=0)
    torch.save(hisvec_list.float(), config["path"]["vec_path"])
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="retrieval_ele.yaml")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--type", type=str, default="encode", choices=["encode", "retrieval"]
    )
    parser.add_argument("--encoder", default="TCN")

    args = parser.parse_args()
    print(args)

    path = "./config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["retrieval"]["encoder"] = args.encoder
    model = TCN(
            input_size=config["retrieval"]["length"],
            output_size=config["retrieval"]["length"], num_channels=[config["retrieval"]["length"]] * (config["retrieval"]["level"]) + [config["retrieval"]["length"]],
        ).to(config["retrieval"]["device"])
    model=torch.load(config["path"]["encoder_path"], map_location='cuda:0')
    if args.type == 'encode':
        all_encode(model,config)
    if args.type == 'retrieval':
        all_retrieval(model,config)


    
