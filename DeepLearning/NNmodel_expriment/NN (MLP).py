'''''
预测模型：简单神经网络— 多层感知机（MLP）
'''''

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.utils import to_categorical

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

# 构建神经网络模型
NN_1 = Sequential()
NN_1.add(Dense(45, input_dim=87, activation='relu'))
NN_1.add(Dropout(0.5))
NN_1.add(Dense(130,activation='relu'))
NN_1.add(Dropout(0.5))
NN_1.add(Dense(2,activation='sigmoid'))

sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9)
NN_1.compile(optimizer=sgd, metrics=['accuracy'], loss='categorical_crossentropy')
NN_1.fit(x=X_train,y=y_train,batch_size=1000,epochs=20)

NN_1_score=NN_1.evaluate(X_test,y_test,batch_size=500)
print("Accuracy of NN_1 : ",NN_1_score)












