from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
import numpy as np
import pandas as pd
import random


class MlpCreator:
    """
    # layer_num: 总层数（包括输入和输出层）
    # nodes：节点数量list（不包括输出层), 长度比层数少1
    # dropout：boolen
    # False：dropout_nums = none;
    # True: dropout_num
    # 包含每一个dropout层的比例，长度比层数少1
    # optimizer 优化器，如果为 “sgd" 则需要 lr,decay,mov 三个参数；否则这三个参数值都为 none
    """
    def __init__(self,layer_num,nodes,dropout,dropout_nums):
        self.layer_num=layer_num
        self.nodes=nodes
        self.dropout = dropout
        self.dropout_nums = dropout_nums
        self.act='softmax'
        self.optimizer='sgd'
        self.lr=0.0
        self.decay=0.0
        self.mov=0.0
        self.accuracy=0.0


    def build_model(self,input_dim,output_dim):
        model = Sequential()
        # 输入层
        model.add(Dense(self.nodes[0], input_dim=input_dim, activation=self.act))
        # 是否使用Dropout，若是，在除了输出层以外的每层后面添加Dropout，每次Dropout的比例按照dropout_nums中的取值依次设定
        if self.dropout:
            model.add(Dropout(self.dropout_nums[0]))
            for i in range(0, self.layer_num - 2):
                model.add(Dense(self.nodes[i + 1], activation=self.act))
                model.add(Dropout(self.dropout_nums[i + 1]))
        else:
            for i in range(0, self.layer_num - 2):
                model.add(Dense(self.nodes[i + 1], activation=self.act))
        # 输出层
        model.add(Dense(output_dim, activation='sigmoid'))

        #优化器
        if self.optimizer=="sgd":
            sgd = optimizers.SGD(lr=self.lr, decay=self.decay, momentum=self.mov)
            model.compile(optimizer=sgd, metrics=['accuracy'], loss='categorical_crossentropy')
        else:
            model.compile(optimizer=self.optimizer, metrics=['accuracy'], loss='categorical_crossentropy')

        return model

    def train_model(self,data,train_batch,train_ephoc,eva_size):
        input_dim=data["X_train"].shape[1]
        output_dim=data["y_train"].shape[1]
        model=self.build_model(input_dim,output_dim)
        model.fit(data["X_train"], data["y_train"], batch_size=train_batch,epochs=train_ephoc)
        score = model.evaluate(data["X_test"], data["y_test"], batch_size=eva_size)
        self.accuracy=score[1]


class MlpSelector:
    """
        layerNum_range：层数选择范围    [int,int,int....]
        nodes_range：每层节点数量选择范围 [min,max]
        dropout：是否采用dropout  Ture/False
        sgd:是否用sgd作优化器 Ture/False
        opt_range: 优化器选择范围    [string,string....]
        act_range：激活函数选择范围  [string,string....]
    """

    def __init__(self, layerNum_range, nodes_range, dropout, sgd, opt_range, act_range):
        self.layerNum_range = layerNum_range
        self.nodes_range = nodes_range
        self.dropout = dropout
        self.sgd = sgd
        self.opt_range = opt_range
        self.act_range = act_range

        """
        默认属性： lrs,decays,movs  sgd优化器相关的三个超参数选择范围
                   train_batch：每个具体模型的批次训练样本数
                   train_epcho：每个具体模型的训练轮数
                   eva_size：评估每个模型用的批次样本数（验证数据）
        """
        self.lrs = [0.01, 0.02, 0.05, 0.07, 0.08]
        self.decays = [1e-6, 3e-6, 5e-6, 7e-6, 1e-5]
        self.movs = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.train_batch = 500
        self.train_epcho = 10
        self.eva_size = 500

        """
        model_features：二维list,每个具体模型的各项参数列表
        scores：每个具体模型的评分
        """
        self.model_features = pd.DataFrame([[0,[],True,[],'act','optimizer',0.0,0.0,0.0]])
        self.scores = [0.0]

    """
    在不同 layer_num,nodes,dropout 设定下，测试模型评分，（opt和act固定）
    """

    def test_diff_layers(self, data, fixed_params, epoch_optTest, epoch):
        layerNums = self.layerNum_range
        dropout_nums = []
        dropout = self.dropout

        # 在每一不同的层数设定下，测试的轮数
        modelNum = (epoch / len(self.act_range)) / epoch_optTest / len(layerNums)

        for layer_num in layerNums:
            k = 0
            while k < modelNum:
                # 在设定的节点数选择范围内，随机生成每层的节点数列表 nodes
                nodes = [np.random.randint(self.nodes_range[0], self.nodes_range[1]) for j in range(0, layer_num - 1)]
                # 根据drop的设定，选择是否采用dropout
                if (self.dropout):
                    dropout_nums = [0.5 for k in range(0, layer_num - 1)]
                    # 用初始参数构造模型
                    model = MlpCreator(layer_num, nodes, True, dropout_nums)
                else:
                    model = MlpCreator(layer_num, nodes, False, dropout_nums)

                # 用固定参数（opt,act) 设定模型其他超参数
                for key in fixed_params:
                    model.key = fixed_params[key]

                # 记录各模型参数
                arr=[[layer_num,nodes,dropout,dropout_nums,
                      fixed_params['act'],fixed_params['optimizer'],fixed_params['lr'],fixed_params['decay'],fixed_params['mov']]]
                self.model_features=self.model_features.append(arr,ignore_index=True)

                # 训练模型，记录各模型评分
                model.train_model(data, self.train_batch, self.train_epcho, self.eva_size)
                self.scores.append(model.accuracy)
                k += 1

    """
    在不同的 opt 设定下，调用test_diff_layers （act固定）
    """

    def test_diff_opts(self, data, act, epoch):
        fixed_params = {'act': act, 'optimizer': '', 'lr': 0.0, 'decay': 0.0, 'mov': 0.0}
        # 如果不用sgd，opt测试的轮数就等于opt_range的长度（多少种opt)
        if (self.sgd == False):
            epcho_optTest = len(self.opt_range)
            for opt in self.opt_range:
                fixed_params['optimizer'] = opt
                self.test_diff_layers(data, fixed_params, epcho_optTest, epoch)
        else:
            # 如果用sgd, opt测试测轮数
            epcho_optTest = len(self.lrs) * len(self.decays) * len(self.movs)
            fixed_params['optimizer'] = 'sgd'
            i = 0
            while i < epcho_optTest:
                # 每一轮测试，lr,decay,mov都要从设定的范围里随机取值
                fixed_params['lr'] = random.choice(self.lrs)
                fixed_params['decay'] = random.choice(self.decays)
                fixed_params['mov'] = random.choice(self.movs)
                self.test_diff_layers(data, fixed_params, epcho_optTest, epoch)
                i+=1

    """
        在不同的 act 设定下，调用test_diff_opts
    """
    def test_diff_mdoels(self, data, epoch):

        # 在每一种act设定下，测试不同的opt设定
        for act in self.act_range:
            self.test_diff_opts(data, act, epoch)

