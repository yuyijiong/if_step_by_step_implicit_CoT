import torch

a = torch.load("./models/Mistral-7B-v0.1-internal-cot/pytorch_model.bin")
#将a中所有key中的 “base_model.” 删除
b = {k.replace("base_model.",""):v for k,v in a.items()}
#将b转化为ordereddict，保存
torch.save(b,"./models/Mistral-7B-v0.1-internal-cot/pytorch_model.bin")