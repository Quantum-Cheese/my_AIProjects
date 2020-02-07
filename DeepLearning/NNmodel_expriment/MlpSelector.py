from model_MLP import MlpSelector
from keras.utils import to_categorical
import numpy as np
from random import choice
import pandas as pd

# 生成随机测试数据
X_train=np.random.randint(0,10,size=[300,5])
y_train=to_categorical([choice([0,1]) for i in range(0,300)])
X_test=np.random.randint(0,10,size=[50,5])
y_test=to_categorical([choice([0,1]) for j in range(0,50)])

# 生成模型训练数据集
data={"X_train":X_train,"y_train":y_train,"X_test":X_test,"y_test":y_test}

# 设定模型超参数选择范围
layerNum_range=[3,4]
nodes_range=[20,100]
dropout=True
sgd=True
opt_range=['sgd']
act_range=['relu','sigmoid']

# 实例化一个 MlpSelector
mlpSelector=MlpSelector(layerNum_range,nodes_range,dropout,sgd,opt_range,act_range)

# 设定sgd 优化器的各项参数选择范围
mlpSelector.lrs = [0.03, 0.05]
mlpSelector.decays = [1e-6, 3e-6]
mlpSelector.movs = [0.3, 0.5]

# 根据数据集的大小，设定 train_batch，train_epcho 和 eva_size
mlpSelector.train_batch=100
mlpSelector.train_epcho=5
mlpSelector.eva_size=50

# 根据朝超参数选择范围的设定，穷举测试不同的超参数组合
mlpSelector.test_diff_mdoels(data,100)

print(mlpSelector.model_features)
# 记录测试数据，保存为csv文件
test_datas=mlpSelector.model_features.drop(axis=0,index=0,inplace=True)
test_datas.columns=['layer_num','nodes','dropout','dropout_nums','act','optimizer','lr','decay','mov']
test_datas['score']=mlpSelector.scores
test_datas.to_csv('MLP_model test results_3.csv',index=False)


