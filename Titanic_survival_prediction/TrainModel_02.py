'''
引入sklearn，尝试不同的算法模型，在验证集上预测，并记录各模型的性能评分
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

model_results=[]
#model_results.columns=['modelName','params','accuracy','F1_score','Fbeta_score']

df=pd.read_csv('train_processed.csv')
X_train,X_test,y_train,y_test=train_test_split(df.drop(['Survived'],axis=1),df['Survived'],test_size = 0.2, random_state = 36)

'''
逻辑回归，直接用默认参数
'''
modelA=LogisticRegression()
modelA.fit(X_train,y_train)
y_pred=modelA.predict(X_test)

acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
fb=fbeta_score(y_test,y_pred,beta=0.5)
model_results.append(['LogisticRegression',{'params':'default'},acc,f1,fb])


'''
SVC，多项式核，用网格搜索选择合适的C参数和degree
'''
params={'C':[0.1,0.3,0.5,0.7],'kernel':['poly'],'degree':[2,3,4]}
scorer=make_scorer(accuracy_score)
grider=GridSearchCV(SVC(),params,scoring=scorer,cv=5)
grider.fit(X_train,y_train)

modelB=grider.best_estimator_
y_pred_B=modelB.predict(X_test)

accB=accuracy_score(y_test,y_pred_B)
f1B=f1_score(y_test,y_pred_B)
fbB=fbeta_score(y_test,y_pred_B,beta=0.5)
print(modelB)
model_results.append(['SVC',{'C':modelB.C,'kernel':'poly','degree':modelB.degree},accB,f1B,fbB])

print(model_results)


