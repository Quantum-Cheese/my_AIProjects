from keras.utils import to_categorical
import pandas as pd
import numpy as np
from model_MLP import MlpModel
from random import choice
import time

dataSets = pd.read_csv("processed_data")

# 分割训练集和测试集，80%的数据用来训练，其余测试
sample_index=np.random.choice(dataSets.index, size=int(len(dataSets)*0.8), replace=False)
trainSets,testSets = dataSets.iloc[sample_index],dataSets.drop(sample_index)

# 区分特征数据和标签数据（输入和输出）,把y转换成binary格式：两类（0或1）
X_train=trainSets.drop(['income'],axis=1)
X_test=testSets.drop(['income'], axis=1)
y_train=to_categorical(trainSets['income'])
y_test=to_categorical(testSets['income'])
# 训练数据 X：（9045，87）y：（9045，2），测试数据 X：（36177，87）y：（36177，2）
data = {"X_train":X_train,"y_train":y_train,"X_test":X_test,"y_test":y_test}


'''
 定义函数：测试不同层数和节点数量的Mlp,其他参数固定
 params: 训练数据 data; 层数：layer_num; 训练模型数量：modelNums; 
        模型其他超参数：fixed_params (调用时需传入固定值）
'''
def test_diff_layers(data,layer_num,modelNums,fixed_params):
    i = 0
    model_features = []
    scores = []
    while i < modelNums:
        # 随机生成每层的节点数列表 nodes
        nodes = [np.random.randint(50, 200) for j in range(0, layer_num - 1)]
        #设定dropout
        dropout=True
        dropout_num=[0.5 for k in range(0,layer_num-1)]

        # 用参数构造模型
        model = MlpModel(layer_num, nodes,dropout,dropout_num)
        for key in fixed_params:
            model.key=fixed_params[key]

        # 记录各模型参数
        model_features.append([])
        model_features[i].append(layer_num)
        model_features[i].append(nodes)
        model_features[i].append(dropout)
        model_features[i].append(dropout_num)
        for element in fixed_params.values():
            model_features[i].append(element)

       # 训练模型，记录各模型评分
        model.train_model(data, 700, 20)
        scores.append(model.accuracy)
        i += 1

    return [model_features,scores]


lrs=[0.01,0.02,0.05,0.07,0.08]
decays=[1e-6,3e-6,5e-6,7e-6,1e-5]
movs=[0.1,0.3,0.5,0.7,0.9]
epoche=0
model_features, scores=[],[]
# 记录开始时间
startTime=time.localtime(time.time())
while epoche<50:
    fixed_params={'act': 'relu', 'optimizer': 'sgd', 'lr': choice(lrs), 'decay': choice(decays), 'mov': choice(movs)}
    layer_nums=[3,4,5,6]
    this_result=test_diff_layers(data,choice(layer_nums),80,fixed_params)
    model_features+=this_result[0]
    scores+=this_result[1]
    epoche+=1

datas=pd.DataFrame(model_features,columns=['layer_num','nodes','dropout','dropout_nums','act','optimizer','lr','decay','mov'])
datas['score']=scores
datas.to_csv('MLP_model test results_0.csv',index=False)
# 记录结束时间
endTime=time.localtime(time.time())

print('Start----{}:{}:{}'.format(startTime.tm_hour,startTime.tm_min,startTime.tm_sec))
print('End-----{}:{}:{}'.format(endTime.tm_hour,endTime.tm_min,endTime.tm_sec))







