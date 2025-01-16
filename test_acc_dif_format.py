import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteriaList, LogitsProcessorList

from tqdm import tqdm
from generate_arithmetic import chain_num


def extract_answer(text):
    try:
        answer_reply = re.search(r'([\d\.-]+)', text).group(1)
        answer_reply = float(answer_reply)
    except:
        answer_reply = None

    return answer_reply


if __name__ == '__main__':

    num_samples = 1000 # the number of samples
    chain_length = 3  # the number of steps of the problem
    internal_cot = True # whether to use internal cot model

    if internal_cot:
        model_path = "./models/Mistral-7B-v0.1-internal-cot"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        from Internalize_CoT_Step_by_Step.src.utils import DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor

        logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(tokenizer.eos_token_id)])
        stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(tokenizer.eos_token_id)])

    else:
        model_path = "./models//Qwen2.5-72B-Instruct"
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

    # try different problem format: reverse or scale
    for reverse, scale in [(False, 1), (True, 1), (False, 0.1)]:

        preds = []
        refs = []
        df = []
        for i in tqdm(range(num_samples)):
            question, inter_result = chain_num(chain_length, reverse=reverse, scale=scale, max_num=19, min_num=0,
                                               chat_format=False)

            if internal_cot:
                # specific answer format for internal cot
                prompt = question + tokenizer.eos_token + tokenizer.eos_token + " ####"

                input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                                      max_length=100000).to("cuda")
                output = model.generate(input_ids=input_ids.input_ids,
                                        attention_mask=input_ids.attention_mask,
                                        do_sample=False,
                                        logits_processor=logits_processor,
                                        stopping_criteria=stopping_criteria,
                                        max_new_tokens=4, )
            else:
                # specific answer format for qwen2
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                                       add_generation_prompt=True)
                prompt = prompt + "A="

                input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True,
                                      max_length=100000).to("cuda")
                output = model.generate(input_ids=input_ids.input_ids,
                                        attention_mask=input_ids.attention_mask,
                                        do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id,
                                        max_new_tokens=4, )

            response = tokenizer.decode(output[0][input_ids.input_ids.size(1):], skip_special_tokens=True).strip()

            preds.append(response)
            refs.append(inter_result[-1])

            df.append(
                {"question": question, "answer": inter_result[-1], "inter_result": inter_result, "pred": response})

        # gather the results
        df = pd.DataFrame(df)

        df.drop_duplicates(subset="inter_result", inplace=True)

        # extract model_answer as float
        df["model_answer"] = df["pred"].apply(extract_answer)
        df = df[df["model_answer"].notnull()]
        df.reset_index(drop=True, inplace=True)

        # calculate accuracy
        df['correct'] = df['answer'] == df['model_answer']
        print("Accuracy: ", df['correct'].mean(), "reverse:", reverse, "scale:", scale)
        print("sample number", len(df))
