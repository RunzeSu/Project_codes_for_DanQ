from sklearn.metrics import roc_auc_score, roc_curve, auc
import h5py
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
import statistics as st


y_pred_cnn = h5py.File('prediction')
pred=np.array(y_pred_cnn['pred'])
testmat = scipy.io.loadmat('test_add_1.mat')
y_test = testmat['testdata']

auc=[]
for i in range(0,919):
    auc.append(roc_auc_score( y_test[:,i], pred[:,i]))
print(auc)
print(sum(auc)/919)

y=range(0,125)
plt.figure()
plt.plot(y,sorted(auc[0:125]))
plt.show()
y=range(0,690)
plt.figure()
plt.plot(y,sorted(auc[125:815]))
plt.show()
y3=range(0,104)
plt.figure()
plt.plot(y3,sorted(auc[815:919]))
plt.show()
y=range(0,919)
y1=range(0,125)
y2=range(125,815)
y3=range(815,919)
plt.figure()
plt.plot(y1,sorted(auc[0:125]))
plt.plot([125,125],[0,1],'r--')
plt.plot([815,815],[0,1],'r--')
plt.plot(y2,sorted(auc[125:815]))
plt.plot(y3,sorted(auc[815:919]))

dictionary1=np.argsort(np.array(auc))
print(dictionary1)

sparsity = []
for i in dictionary1:
    sparsity.append(sum(y_test[:,i]))
print(sparsity)
y=range(0,919)
plt.figure()
plt.scatter(auc,sparsity)

precision, recall, th = precision_recall_curve(y_test[:,i], pred[:,i])
plt.plot(recall,precision)
#plt.plot(recall[260600:260699],precision[260600:260699])
plt.show()

i=193 
precision, recall, _ = precision_recall_curve(y_test[:,i], pred[:,i])
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'PR plot of the A549 USF1')

pr_auc1=[]
from sklearn.metrics import auc
for i in range(0,919):
    precision, recall, th = precision_recall_curve(y_test[:,i], pred[:,i])
    pr_auc1.append(auc( recall, precision))
print(pr_auc1)

st.mean(pr_auc1)
y=range(0,919)
plt.figure()
plt.plot(y,sorted(np.array(pr_auc1)))
plt.xlabel('PR-AUC')
plt.ylabel('Orders')
plt.title('PR-AUC from low to high')
plt.show()

y=range(0,918)
y1=range(0,125)
y2=range(125,814)
y3=range(814,918)
plt.figure()
plt.plot(y1,sorted(pr_auc1[0:125]))
plt.plot([125,125],[0,1],'r--')
plt.plot([814,814],[0,1],'r--')
plt.plot(y2,sorted(pr_auc1[125:125+689]))
plt.plot(y3,sorted(pr_auc1[918-104:918]))
plt.ylabel('PR-AUC')
plt.xlabel('Orders')
plt.title('PR-AUC from low to high in different responses')
plt.show()

dictionary1=np.argsort(np.array(pr_auc1))
sparsity = []
for i in dictionary1:
    sparsity.append(sum(y_test[:,i]))
print(sparsity)

y=range(0,919)
plt.figure()
plt.plot(y,sparsity)
plt.xlabel('indexes')
plt.ylabel('Number of 1 within a response')
plt.show()

y=range(0,919)
plt.figure()
plt.scatter(pr_auc1,sparsity)
plt.xlabel('PR_ROC')
plt.ylabel('Number of 1 within a response')
plt.show()
