import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#y_true = np.load('truth.npy')
#y_pred = np.load('result.npy')
result = np.load('result.npz')
y_true = result['true']
y_pred = result['pred']
#y_pred[y_pred > 5] = 0
y_pred = np.clip(y_pred, None, 5)

label_names = ['CON','GGO','HCM','EMP','NOR','Others']
labels = range(1,7)

# %% calculate true positive rate
tp_rates = []
for label, name in zip(labels, label_names):
    mask = y_true == label
    t = y_true[mask]
    p = y_pred[mask]
    tp = np.sum(t==p)
    tp_rate = tp/t.size
    print('recall', name, t.size, tp, tp_rate)
    mask = y_pred == label
    t = y_true[mask]
    p = y_pred[mask]
    tp = np.sum(t==p)
    precision = tp / t.size
    print('precision', name, t.size, tp, precision)
    tp_rates.append(tp_rate)
print('mean', np.mean(tp_rates))

# %%

cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
cmx_data = cmx_data[:,:-1]
cmx_data = 100 * cmx_data / cmx_data.sum(axis=0)[np.newaxis,:]

df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels[:-1])
df_cmx.index = label_names 
df_cmx.columns = label_names[:-1]
#df_norm_col=(df-df.mean())/df.std()

# %%
plt.figure(figsize = (10,10))
sns.set(context='notebook',style="whitegrid",font='serif',font_scale=2)
sns.heatmap(df_cmx, annot=True, fmt = '.1f',cmap='Reds', vmin=0, vmax=100, square=True)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.yticks(rotation=90,va='center') 
plt.savefig('cm.eps', bbox_inches='tight')