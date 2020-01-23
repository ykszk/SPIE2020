import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

labels = [('CON',1),
          ('GGO',2),
          ('HCM',3),
          ('EMP',4),
          ('NOR',5)]
columns = []
for name, i in labels:
    columns.append(pd.read_csv('{}.csv'.format(i), index_col=0)['Dice'])

# %%
all_labels = sum([[name]*len(column) for (name,_), column in zip(labels, columns)], [])
all_data = sum([list(c) for c in columns], [])
df_src = [(t,l) for t,l in zip(all_data,all_labels)]
df = pd.DataFrame(df_src, columns=['dice','label'])
# %%
plt.figure(figsize=(10,4))
sns.set(context='notebook',style="whitegrid",font='serif',font_scale=1.5)
dld_palette = sns.color_palette(["#00ffff",'#ffff00','#ff0000','#00ff00','#994c00'])
sns.set_palette(dld_palette)

ax = sns.swarmplot(x='dice', y ='label', data=df, linewidth=.3, s=3)#,hue='label',palette=dld_palette)
ax = sns.boxenplot(x='dice', y='label', data=df, color="0.8")
plt.ylabel('')
plt.xlabel('')
plt.xlim((-0.02,1.02))
plt.savefig('plot.eps', bbox_inches='tight')