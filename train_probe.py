# Encoding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from tqdm import tqdm
import pandas as pd
from transformers import TrainingArguments, Trainer
from torch.nn import CrossEntropyLoss
import json
import pathlib
from datasets import Dataset


def data_collator(features):
    # features包含hidden_states和labels
    batch = {}
    batch["hidden_states"] = torch.tensor([feature["hidden_states"] for feature in features]).float().cuda()
    batch["labels"] = torch.tensor([feature["labels"] for feature in features]).long().cuda()
    return batch


class LinearProbing(torch.nn.Module):
    def __init__(self, hidden_size, output_dim):
        super(LinearProbing, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, hidden_states, labels=None):
        logits = self.linear2(self.linear1(hidden_states))
        # logits = self.linear2(hidden_states)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits, "loss": None}


def evaluate(model, eval_dataset):
    model.eval()
    # 对每一个数据，linear给出分类结果，计算准确率
    pred_list = []

    for i in tqdm(range(len(eval_dataset))):
        hidden_states = data_collator([eval_dataset[i]])["hidden_states"]
        with torch.no_grad():
            logits = model(hidden_states)["logits"]
        pred = torch.argmax(logits[0], dim=-1).item()
        pred_list.append(pred)

    # 计算准确率
    labels = [int(eval_dataset[i]["labels"]) for i in range(len(eval_dataset))]
    correct_num = sum([1 for i in range(len(labels)) if labels[i] == pred_list[i]])
    acc = correct_num / len(labels)
    return acc


if __name__ == '__main__':

    output_dir = "./linear_probing"
    print(output_dir)

    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        eval_steps=10000,
        report_to="tensorboard",
        logging_strategy='steps',
        logging_steps=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        save_strategy='steps',
        save_steps=10000,
        num_train_epochs=8,
        remove_unused_columns=False,
        ignore_data_skip=True,
        save_only_model=True,

        optim="adamw_torch",
        weight_decay=0,

        lr_scheduler_type="constant_with_warmup",  # "linear",  #
        warmup_ratio=0.05,
        learning_rate=1e-4,
        per_device_train_batch_size=200,
        max_grad_norm=1.0,

        # max_steps=1,
        auto_find_batch_size=False,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,

        seed=1,
    )

    probing_dataset_path = "./hidden_state_result/Mistral-7B-v0.1-internal-cot_chain_len=5.json"
    hidden_size = 4096  # 5120#3584#1536
    num_layers = 32

    output_dim = 20
    only_choose_correct = False  # 是否只保留答对的样本

    # 提取probing_dataset_path中chain=后面的数字
    chain_num = int(probing_dataset_path.split("=")[1][0])

    print("Load data: ", probing_dataset_path)
    df = pd.read_json(probing_dataset_path, orient="records", lines=True, dtype=False)

    save_path = os.path.basename(probing_dataset_path).replace(".json", "_每一步.json")
    save_path = "./probing_results/" + save_path
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    # 看准确率
    acc_all = df["correct"].mean()
    print("Accuracy ", acc_all)
    print("Sample Num ", len(df))

    # 只保留correct为True的
    if only_choose_correct:
        df = df[df["correct"] == True]
        df.reset_index(drop=True, inplace=True)

    # for each intermediate result, train a linear classifier
    results_per_step = {}

    for step_chosen in range(chain_num):

        # choose this step's label
        df["labels"] = df["inter_result"].apply(lambda x: x[step_chosen])

        # 将hidden_states_last_token重命名为hidden_states
        if "hidden_states_last_token" in df.columns:
            df.rename(columns={"hidden_states_last_token": "hidden_states"}, inplace=True)
        if "hidden_state" in df.columns:
            df.rename(columns={"hidden_state": "hidden_states"}, inplace=True)

        # hidden_states的shape为(num_layers,num_tokens,hidden_size)

        # 将hidden_states分别取不同的层，形成32个df
        results = {}
        for layer in range(num_layers - 1, -1, -1):  # range(-16,0,1):#
            print("layer:", layer)
            df_layer = df.copy()
            df_layer["hidden_states"] = df_layer["hidden_states"].apply(lambda x: x[layer])
            dataset = Dataset.from_pandas(df_layer)

            # 切分测试集
            dataset_dict = dataset.train_test_split(test_size=0.2, seed=1)
            train_dataset = dataset_dict["train"]
            test_dataset = dataset_dict["test"]

            print("训练样本数：", len(train_dataset))

            # 初始化一个torch模型，一层线性层
            model = LinearProbing(hidden_size, output_dim)
            model.requires_grad_()
            # 初始化一个trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
            )
            # 开始训练
            trainer.train()

            # 评估
            acc = evaluate(model, test_dataset)
            print("layer:", layer, "acc:", acc)
            results[str(layer)] = round(acc, 4)

        print(results)
        results_per_step["step {}/{}".format(str(step_chosen + 1), str(chain_num))] = results

    with open(save_path, "w") as f:
        json.dump(results_per_step, f)
    print(results_per_step)
    print("save to:", save_path)
