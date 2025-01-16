import random
from 解决逻辑检索.api回复 import model_answer
import re
from tqdm import tqdm
def chain_num(length,reverse=False,scale=None,max_num=19,min_num=0,chat_format=True):

    #随机生成length个min_num到max_num之间的整数，作为inter_result
    inter_result=[random.randint(min_num,max_num) for i in range(length)]

    #根据inter_result的差值计算加数
    addend=[inter_result[i]-inter_result[i-1] for i in range(1,length)]
    addend.insert(0,inter_result[0])


    if scale is not None:
        addend=[round(i*scale,1) for i in addend]
        inter_result=[round(i*scale,1) for i in inter_result]

    #转化为字符串
    addend_str=[" + "+str(i) if i>0 else " - "+str(-i) for i in addend]

    variable_list="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    #生成问题
    equation_list=[]
    #首先生成第一个变量的值
    equation_list.append(f"{variable_list[length-1]} = {addend[0]} ;")
    for i in range(1,length):
        equation_list.append(f"{variable_list[length-i-1]} = {variable_list[length-i]}{addend_str[i]} ;")

    if reverse:
        equation_list=equation_list[::-1]

    if chat_format:
        question="\n".join(equation_list)+f"\n\nQuestion: What is the value of A? You must answer directly with A=xxx."
    else:
        question = "\n".join(equation_list) + f"\n\nA=?"

    return question,tuple(inter_result)


if __name__ == '__main__':
    num_samples=100
    chain_length=5
    model_name="qwen2.5-7b"

    df=[]

    for i in tqdm(range(num_samples)):
        question,inter_result=chain_num(chain_length,reverse=False,scale=1)

        reply=model_answer(question,model_name,temperature=0,base_url="http://localhost:5004/v1")

        #reply中提取答案，即A=后面的数字，数字可能是负数
        try:
            answer_reply=re.findall(r'A\s*=\s*([\d-]+)',reply)[-1]
            answer_reply=float(answer_reply)
        except:
            answer_reply=None

        df.append({"question":question,"answer":inter_result[-1],"model_answer":answer_reply,"reply":reply})
        if i<3:
            print(answer_reply)
    import pandas as pd
    df=pd.DataFrame(df)

    #计算准确率
    df['correct']=df['answer']==df['model_answer']
    print(df['correct'].mean())

    print("chain_length:",chain_length)
