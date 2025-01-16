import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Qwen2ForCausalLM, GPT2LMHeadModel
from transformers import StoppingCriteriaList, LogitsProcessorList

from tqdm import tqdm
from generate_arithmetic import chain_num


def get_hidden_state(texts, model: Qwen2ForCausalLM, tokenizer, layer: int = None, generate_tokens: int = 1,
                     compress_group_size=None, internal_cot=False):
    hidden_states_last_token_list = []
    pred_list = []
    # 分别对每个text进行编码
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=100000).to(
            model.device)

        input_ids = inputs.input_ids
        input_len = input_ids.size(1)
        with torch.no_grad():
            if internal_cot:
                output = model.generate(input_ids=input_ids,
                                        do_sample=False,
                                        logits_processor=logits_processor,
                                        stopping_criteria=stopping_criteria,
                                        max_new_tokens=generate_tokens,
                                        output_hidden_states=True,
                                        return_dict_in_generate=True, )
            else:
                output = model.generate(input_ids=input_ids,
                                        max_new_tokens=generate_tokens,
                                        do_sample=False,
                                        output_hidden_states=True,
                                        return_dict_in_generate=True, )

        pred_str = tokenizer.decode(output["sequences"][0][input_len:], skip_special_tokens=True).strip()

        # 获取prefill的hidden_states
        hidden_states = output.hidden_states[0]
        # 获取每一层的hidden_state的最后一个token
        if layer is None:
            hidden_states_last_token = [hidden_state[0, -1] for hidden_state in hidden_states[1:]]  # 不包括第一层
            # 如果compress_group_size不为None，对hidden_states_last_token进行压缩。每compress_group_size个hidden_state取平均
            if compress_group_size is not None:
                hidden_states_last_token = [torch.stack(hidden_states_last_token[i:i + compress_group_size]).mean(0) for
                                            i in range(0, len(hidden_states_last_token), compress_group_size)]

            # 将hidden_states_last_token转化为list
            hidden_states_last_token = [hidden_state.tolist() for hidden_state in hidden_states_last_token]
        else:
            hidden_states_last_token = hidden_states[layer + 1][0, -1].tolist()

        hidden_states_last_token_list.append(hidden_states_last_token)

        pred_list.append(pred_str)
        print("pred:", pred_str)

    return hidden_states_last_token_list, pred_list


def extract_answer(text):
    try:
        answer_reply = re.search(r'([\d\.-]+)', text).group(1)
        answer_reply = float(answer_reply)
    except:
        answer_reply = None

    return answer_reply


if __name__ == '__main__':

    num_samples = 2000
    chain_length = 4
    internal_cot = False

    if internal_cot:
        model_path = "./models/Mistral-7B-v0.1-internal-cot"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        from Internalize_CoT_Step_by_Step.src.utils import DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor

        logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(tokenizer.eos_token_id)])
        stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(tokenizer.eos_token_id)])

    else:
        model_path = "./models/Qwen2.5-72B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=True,
                                                     # attn_implementation="flash_attention_2",  # "eager",#
                                                     device_map="auto",
                                                     )

    # 生成num_samples个样本
    question_and_inter_result_list = [
        chain_num(chain_length, reverse=False, scale=1, max_num=19, min_num=0, chat_format=False) for i in
        range(num_samples * 4)]
    # 删除重复的inter_result
    question_and_inter_result_list = list(set(question_and_inter_result_list))
    # 取前num_samples个
    question_and_inter_result_list = question_and_inter_result_list[:num_samples]

    df = []

    for i in tqdm(range(len(question_and_inter_result_list))):
        question, inter_result = question_and_inter_result_list[i]

        if internal_cot:
            prompt = question + tokenizer.eos_token + tokenizer.eos_token + " ####"

            hidden_states_last_token_list, pred_list = get_hidden_state([prompt], model, tokenizer, layer=None,
                                                                        generate_tokens=4, compress_group_size=None,
                                                                        internal_cot=internal_cot)
        else:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                                   add_generation_prompt=True)
            prompt = prompt + "A="
            hidden_states_last_token_list, pred_list = get_hidden_state([prompt], model, tokenizer, layer=None,
                                                                        generate_tokens=4, compress_group_size=2)
        if i == 0:
            print(question, pred_list[0])

        df.append({"question": question, "answer": inter_result[-1], "inter_result": inter_result,
                   "hidden_states": hidden_states_last_token_list[0], "pred": pred_list[0]})

    # gather the results
    df = pd.DataFrame(df)

    df.drop_duplicates(subset="inter_result", inplace=True)

    # extract model_answer as float
    df["model_answer"] = df["pred"].apply(extract_answer)
    df = df[df["model_answer"].notnull()]
    df.reset_index(drop=True, inplace=True)

    # calculate accuracy
    df['correct'] = df['answer'] == df['model_answer']
    print("Accuracy: ", df['correct'].mean())
    print("sample number", len(df))

    # save hidden states
    model_name = os.path.basename(model_path)
    save_path = "./hidden_state_result/{}_chain_len={}.json".format(model_name, chain_length)
    import pathlib

    pathlib.Path("./hidden_state_result").mkdir(exist_ok=True)
    df.to_json(save_path, orient="records", lines=True)
    print("hidden size", model.config.hidden_size)
    print("save to", save_path)
