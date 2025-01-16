import matplotlib.pyplot as plt
import json

full_data= \
    {'step 1/4': {'31': 0.9675, '30': 0.98, '29': 0.9775, '28': 0.98, '27': 0.9825, '26': 0.985, '25': 0.985,
                  '24': 0.9825, '23': 0.9775, '22': 0.9725, '21': 0.9625, '20': 0.95, '19': 0.93, '18': 0.95,
                  '17': 0.9125, '16': 0.6475, '15': 0.3225, '14': 0.365, '13': 0.3375, '12': 0.32, '11': 0.3575,
                  '10': 0.4125, '9': 0.43, '8': 0.4575, '7': 0.335, '6': 0.355, '5': 0.385, '4': 0.3925, '3': 0.1125,
                  '2': 0.09, '1': 0.08, '0': 0.055},
     'step 2/4': {'31': 0.29, '30': 0.275, '29': 0.3, '28': 0.29, '27': 0.265, '26': 0.2675, '25': 0.265, '24': 0.25,
                  '23': 0.2275, '22': 0.235, '21': 0.2275, '20': 0.2325, '19': 0.22, '18': 0.225, '17': 0.21,
                  '16': 0.1725, '15': 0.165, '14': 0.1625, '13': 0.1525, '12': 0.1325, '11': 0.135, '10': 0.1225,
                  '9': 0.1, '8': 0.1125, '7': 0.0725, '6': 0.0625, '5': 0.0625, '4': 0.055, '3': 0.055, '2': 0.055,
                  '1': 0.055, '0': 0.055},
     'step 3/4': {'31': 0.1875, '30': 0.185, '29': 0.1675, '28': 0.165, '27': 0.1825, '26': 0.19, '25': 0.1875,
                  '24': 0.1775, '23': 0.1625, '22': 0.1725, '21': 0.155, '20': 0.1525, '19': 0.1575, '18': 0.1475,
                  '17': 0.1425, '16': 0.1325, '15': 0.14, '14': 0.1525, '13': 0.16, '12': 0.145, '11': 0.14, '10': 0.12,
                  '9': 0.1125, '8': 0.1025, '7': 0.0625, '6': 0.0575, '5': 0.07, '4': 0.0675, '3': 0.0675, '2': 0.0675,
                  '1': 0.0675, '0': 0.0675},
     'step 4/4': {'31': 0.165, '30': 0.1375, '29': 0.125, '28': 0.1125, '27': 0.1, '26': 0.0975, '25': 0.1075,
                  '24': 0.0975, '23': 0.1125, '22': 0.115, '21': 0.115, '20': 0.1175, '19': 0.13, '18': 0.1225,
                  '17': 0.1275, '16': 0.1325, '15': 0.1425, '14': 0.14, '13': 0.145, '12': 0.125, '11': 0.1175,
                  '10': 0.1275, '9': 0.12, '8': 0.085, '7': 0.095, '6': 0.085, '5': 0.0825, '4': 0.08, '3': 0.0825,
                  '2': 0.07, '1': 0.0675, '0': 0.0675}}

file_name="4step_mistral"

plt.figure(figsize=(10, 7))
num_layers=len(full_data[list(full_data.keys())[0]])
for key,data in full_data.items():

    # 按key排序
    data = sorted(data.items(), key=lambda x: int(x[0]))

    y_data = [x[1] for x in data]
    #所有数值乘以100
    y_data = [x*100 for x in y_data]

    # 绘制图像,x轴为key，y轴为value
    plt.plot([x[0] for x in data], y_data,label=key)



plt.ylim(0, 101)
#plt.xticks(range(len(data)), fontsize=20)
plt.xticks(range(1, num_layers, 2), fontsize=20)
plt.yticks(fontsize=20)
#plt.xticks(fontsize=12)
plt.xlabel("Layer", fontsize=25)
plt.ylabel("Accuracy (%)", fontsize=25)

plt.tight_layout()
plt.rcParams.update({'font.size': 20})
plt.legend(loc='upper left')
plt.savefig('./{}.png'.format(file_name))
#plt.show()
