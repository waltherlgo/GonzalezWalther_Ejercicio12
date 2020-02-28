import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
target=target==1
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
Modelo=[0,1,2]
Titulos=['PCA sobre 0','PCA sobre 1','PCA sobre todos']
plt.figure(figsize=(20,10))
for i in Modelo:
    if (i==2):
        dd=np.arange(x_train.shape[0])
    else:
        if (i==1):
            dd = y_train==1
        else:
            dd = y_train!=1
    cov = np.cov(x_train[dd].T)
    valores, vectores = np.linalg.eig(cov)
    valores = np.real(valores)
    vectores = np.real(vectores)
    ii = np.argsort(-valores)
    valores = valores[ii]
    vectores = vectores[:,ii]
    Data_train=x_train@vectores.T
    Data_test=x_test@vectores.T
    #for model in Modelos:
    #    clf = LinearDiscriminantAnalysis(solver=model)
    Para=10
    clf = LinearDiscriminantAnalysis()
    clf.fit(Data_train[:,:Para], y_train)
    y_pred=clf.predict_proba(Data_test[:,:Para])
    precision, recall, thresholds = precision_recall_curve(y_test,y_pred[:,1])
    F1=2*precision*recall/(precision+recall)
    F1=np.nan_to_num(F1)
    plt.subplot(1,2,2)
    plt.scatter(recall[np.argmax(F1)],precision[np.argmax(F1)],color='r')
    plt.plot(recall,precision)
    plt.subplot(1,2,1)
    plt.scatter(thresholds[np.argmax(F1)],np.max(F1),color='r')
    plt.plot(thresholds,F1[1:])
plt.subplot(1,2,2)
plt.xlabel("Cobertura")
plt.ylabel("Precision")
plt.legend(Titulos)
plt.subplot(1,2,1)
plt.xlabel("Probabilidad")
plt.ylabel("F1")
plt.legend(Titulos)   
plt.savefig('F1_prec_recall.png')
plt.show()