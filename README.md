# Do LLMs Really Think Step-by-step In Implicit Reasoning?

## Paper
[Do LLMs Really Think Step-by-step In Implicit Reasoning?](https://arxiv.org/abs/2411.15862)

## Model used

``models/Qwen2.5-72B-Instruct``: Qwen2.5 model, the model weights can be downloaded from huggingface.

``models/Mistral-7B-v0.1-internal-cot``: Specialized trained model based on mistral. 
The training method is [here](https://github.com/da03/Internalize_CoT_Step_by_Step) and the model weights can be downloaded [here](https://drive.google.com/drive/folders/1azfzWxf2jy1H7XAe-dAhtYFTPuh7tfmd).
The downloaded model weights need to be converted to huggingface format using ``change_ckpt_format.py``

## Code

### generate data
``generate_arithmetic.py``: Generate random arithmetic problems for the experiment.

### EXP1: probe intermediate results when using implicit thinking
``get_hidden_states.py``: Getting the hidden states of the model when handling arithmetic problems.

``train_probe.py``: Train and test a linear probe for each layer of each intermediate result.

``visualize_probe_acc.py``: Visualize the probe accuracy of each layer of each intermediate result.

### EXP2: test the model's accuracy when the format changes when using implicit thinking
``test_acc_dif_format.py``: Test the model's accuracy in arithmetic problems under different formats.



