'''
训练模型：多层感知机（神经网络）
引入模块 model_MPL
先预先定义好的mlp模板（MlpCreator），快速生成模型
设定超参数选择范围，用最优参数选择器（MlpSelector）进行参数调优，记录下所有参数组合和准确率数据
选出模型准确率最高的一组参数，用最优超参数，评估模型其他性能，并在测试上做出预测
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from model_MLP import MlpSelector

# 导入数据并处理
df=pd.read_csv('train_processed.csv')
X=df.drop('Survived',axis=1)
y=to_categorical(df['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 36)

print(y_test.shape)
# 生成模型训练数据集
data={"X_train":X_train,"y_train":y_train,"X_test":X_test,"y_test":y_test}


# 设定模型超参数选择范围
layerNum_range=[3,4]
nodes_range=[50,120]
dropout=False
sgd=True
opt_range=['sgd']
act_range=['relu','sigmoid']

# 实例化一个MLP模型
mlpSelector=MlpSelector(layerNum_range,nodes_range,dropout,sgd,opt_range,act_range)

# 设定sgd 优化器的各项参数选择范围
mlpSelector.lrs = [0.01, 0.03]
mlpSelector.decays = [1e-6, 3e-6]
mlpSelector.movs = [0.7, 0.9]

# 根据数据集的大小，设定 train_batch，train_epcho 和 eva_size
mlpSelector.train_batch=200
mlpSelector.train_epcho=5
mlpSelector.eva_size=150

# 根据超参数选择范围的设定，穷举测试不同的超参数组合
mlpSelector.test_diff_mdoels(data,100)

# 记录测试数据，保存为csv文件
results=mlpSelector.model_features
results.columns=['layer_num','nodes','dropout','dropout_nums','act','optimizer','lr','decay','mov']
results['score']=mlpSelector.scores
results.to_csv('MLP_model test results_1.csv',index=False)
