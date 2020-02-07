from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers

"""

构造模型参数字典的函数
layer_num:总层数（包括输入和输出层）
nodes：节点数量list（不包括输出层),长度比层数少1
dropout：boolen False：dropout_nums=[];True: dropout_num 包含每一个dropout层的比例，长度比层数少1
sgd：boolen，True：设置lr，decay，momentum，且other_opt为none；False:这几项都设为0.0,other_opt需设定（String类型）
"""
def set_params(layer_num,nodes,act,dropout,dropout_nums,other_opt,sgd,lr,decay,mov):
    params={"layer_num":layer_num,
            "nodes":nodes,
            "act":act,
            "dropout":dropout,
            "dropout_nums":dropout_nums,
            "other_opt":other_opt,
            "sgd":sgd,
            "lr":lr,
            "decay":decay,
            "mov":mov
            }
    return params

"""
构建NN模型的模板函数
params：通过 set_params()构造的参数字典
"""
def build_NN(input_dim,output_dim,params):
    model=Sequential()
    # 输入层
    model.add(Dense(params["nodes"][0],input_dim=input_dim,activation=params["act"]))
    # 是否使用Dropout，若是，在除了输出层以外的每层后面添加Dropout，每次Dropout的比例按照dropout_nums中的取值依次设定
    if params["dropout"]:
        model.add(Dropout(params["dropout_nums"][0]))
        for i in range(0, params["layer_num"] - 2):
            model.add(Dense(params["nodes"][i + 1], activation=params["act"]))
            model.add(Dropout(params["dropout_nums"][i+1]))
    else:
        for i in range(0, params["layer_num"] - 2):
            model.add(Dense(params["nodes"][i + 1], activation=params["act"]))
    # 输出层
    model.add(Dense(output_dim, activation='sigmoid'))

    if params["sgd"]:
        sgd=optimizers.SGD(lr=params["lr"],decay=params["decay"],momentum=params["mov"])
        model.compile(optimizer=sgd, metrics=['accuracy'], loss='categorical_crossentropy')
    else:
        model.compile(optimizer=params["other_opt"], metrics=['accuracy'], loss='categorical_crossentropy')

    return model

"""
训练模型的函数，训练对象为由build_NN()构建的模型
model: 编译好的模型对象
data:训练和测试数据
batch：批次样本数
epoc：训练周期数
"""
def train_model(model,data,batch,epoc):
    model.fit(data["X_train"],data["y_train"],batch_size=batch,epochs=epoc)
    score=model.evaluate(data["X_test"],data["y_test"],batch_size=500)
    return score


# def train_model(model,X_train,y_train,X_test,y_test,batch,epoc):
#     # model=build_NN(X_train.shape[1],y_train.shape[1],model_params)
#     model.fit(x=X_train, y=y_train, batch_size=batch, epochs=epoc)
#     score = model.evaluate(X_test, y_test, batch_size=500)
#     return score





