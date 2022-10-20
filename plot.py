# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:03:38 2020

# Bar Charts and Plots

@author: Parisa
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pylab as plb
import matplotlib.pyplot as mpl
import matplotlib.axes as Axes
import pandas as pd
mpl.rcParams.update(mpl.rcParamsDefault)

print(plt.style.available)
# plt.style.use('bmh')


font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 14,
        }
plt.rcParams.update({'font.size': 14, 'font.family':'sans-serif'})


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])
RF = (88.49, 95.54, 91.0)
FNN = (87.52, 95.26, 90.5)
K_Means = (63, 72, 70)

ind = np.arange(len(RF))
width = 0.17  # the width of the bars
factor = 1.2
ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])
ax.grid(True, color='k', linestyle='--', axis='y', linewidth=0.5, alpha=0.3)

rects1 = ax.bar(ind - factor* width, RF, width,
                color='#FFA07A', label='RF', alpha=1, edgecolor='k',)
rects2 = ax.bar(ind , FNN, width, 
                color='IndianRed', label='FNN', edgecolor='k')
rects3 = ax.bar(ind + factor* width, K_Means, width, 
                color='#1F618D', label='K-Means', edgecolor='k') #5a7d9a #1F618D #adad3b
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([55, 100])
ax.set_xlabel('Input Variables')
ax.set_xticks(ind)
ax.set_xticklabels(('1D', '14D', 'VGG16'))
ax.set_axisbelow(True)
ax.legend()
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 1.25
fig.savefig('Marcelous.jpg', dpi=300, bbox_inches="tight")
plt.show()
##################################################################################
##################################################################################
##################################################################################New
# plt.style.use('bmh')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 14, 'font.family':'sans-serif'})

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])

group_names = ["RF", "FNN", "K_Means", "U_Net"]

RF = (90.56, 95.55, 93.26)
FNN = (90.55, 94.5, 92.0)
K_Means = (60.53, 70.1, 68)
U_Net = (94)

ind = np.arange(len(RF))
width = 0.2  # the width of the bars
factor = 1.2
# ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])
ax.grid(True, color='k', linestyle='--', axis='y', linewidth=0.5, alpha=0.3)

rects1 = ax.bar(ind - factor* width, RF, width,
                color='#FFA07A', label='RF', alpha=1, edgecolor='k',)
rects2 = ax.bar(ind , FNN, width, 
                color='IndianRed', label='FNN', edgecolor='k',)

rects3 = ax.bar(ind + factor* width, K_Means, width, 
                color='#1F618D', label='K-Means',edgecolor='k',) #5a7d9a #1F618D #adad3b
rects4 = ax.bar( 3, U_Net, width, 
                color='#EFA0A0', label='U_Net',edgecolor='k',) #5a7d9a #1F618D #adad3b


ax.set_ylabel('Accuracy (%)')
ax.set_ylim([55, 100])
ax.set_xlabel('Input Variables')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(('Original Image', '14 Features', 'VGG16 Features', 'U_Net Features'))
ax.set_axisbelow(True)
ax.legend(fontsize=12)
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 1.25
plt.show()

fig.savefig('Mancus.jpg', dpi=300, bbox_inches="tight")

##################################################################################New
# plt.style.use('bmh')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 14, 'font.family':'sans-serif'})

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])

group_names = ["RF", "FNN", "K_Means", "U_Net"]

#ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])
RF = (88.49, 95.54, 91.0)
FNN = (87.52, 95.26, 90.5)
K_Means = (63, 72, 70)
U_Net = (92)

ind = np.arange(len(RF))
width = 0.2  # the width of the bars
factor = 1.2
# ax = fig.add_axes([0.1, 0.1, 1.1, 1.2])
ax.grid(True, color='k', linestyle='--', axis='y', linewidth=0.5, alpha=0.3)

rects1 = ax.bar(ind - factor* width, RF, width,
                color='#FFA07A', label='RF', alpha=1, edgecolor='k',)
rects2 = ax.bar(ind , FNN, width, 
                color='IndianRed', label='FNN', edgecolor='k',)

rects3 = ax.bar(ind + factor* width, K_Means, width, 
                color='#1F618D', label='K-Means',edgecolor='k',) #5a7d9a #1F618D #adad3b
rects4 = ax.bar( 3, U_Net, width, 
                color='#EFA0A0', label='U_Net',edgecolor='k',) #5a7d9a #1F618D #adad3b


ax.set_ylabel('Accuracy (%)')
ax.set_ylim([55, 100])
ax.set_xlabel('Input Variables')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(('Original Image', '14 Features', 'VGG16 Features', 'U_Net Features'))
ax.set_axisbelow(True)
ax.legend(fontsize=12)
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 1.25
fig.savefig('Marcelous.jpg', dpi=300, bbox_inches="tight")
plt.show()

##################################################################################
##################################################################################
##################################################################################

df_64 = pd.read_clipboard()
df_64
df_14 = pd.read_clipboard()
df_14.to_csv("df_14.csv", index=False)
df_64.to_csv("df_64.csv", index=False)


df_14 = pd.read_csv("df_14.csv")
df_64 = pd.read_csv("df_64.csv")

mpl.rcParams.update(mpl.rcParamsDefault)
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10,
        }
# plt.style.use('bmh')
plt.rcParams.update({'font.size': 10, 'font.family':'sans-serif'})
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 0.5
y_pos = np.arange(len(df_14)) 

fig = plt.figure()
ax = fig.add_axes([0.5, 0.5, 2.1, 1.2])
#ax.barh(y_pos, df_14.iloc[:, 1], color="#530B13")

ax = df_14.plot.barh(grid=None, legend =False, width=0.7,color="#530B13", alpha=1)
ax.set_axisbelow(True)
ax.grid(True, color='k', linestyle='--', axis='x', linewidth=0.5, alpha=0.3)
ax.set_yticklabels(list(df_14.iloc[:, 0]))
ax.set_xlabel("Feature Importance ")
ax.ticklabel_format(axis="x", style='sci', scilimits=(-2,-2))
plt.gca().invert_yaxis()
fig = ax.get_figure()
fig.savefig('barplots3.jpg', dpi=300, bbox_inches="tight")






mpl.rcParams.update(mpl.rcParamsDefault)

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10,
        }
# plt.style.use('bmh')
plt.rcParams.update({'font.size': 10, 'font.family':'sans-serif'})
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 0.5
y_pos = np.arange(len(df_64))
ax = df_64.plot.barh(grid=None, legend =False, width=0.7, color="#530B13", alpha=1)
ax.set_axisbelow(True)
ax.grid(True, color='k', linestyle='--', axis='both', linewidth=0.5, alpha=0.3)
ax.set_yticklabels(df_64.iloc[:, 0])
ax.set_xlabel("Feature Importance ")
ax.ticklabel_format(axis="x", style='sci', scilimits=(-2,-2))
plt.gca().invert_yaxis()

plt.show()
fig = ax.get_figure()

fig.savefig('barplots4.jpg', dpi=300, bbox_inches="tight")


##################################################################################
###################### only 4 important ##########################################
##################################################################################
### https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

df_14 = pd.read_csv("df_14.csv")
df_64 = pd.read_csv("df_64.csv")
df_14 = df_14.iloc[0:4, :]

mpl.rcParams.update(mpl.rcParamsDefault)
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 10,
        }
# plt.style.use('bmh')
plt.rcParams.update({'font.size': 11, 'font.family':'sans-serif'})
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.linewidth"]  = 0.6

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.5, 0.4])
ax.barh(df_14.iloc[:, 0], df_14.iloc[:, 1], height=0.7, color="#530B13")
ax.set_axisbelow(True)
ax.grid(True, color='k', linestyle='--', axis='x', linewidth=0.3, alpha=0.3)
ax.set_xlabel("Feature Importance ",)
ax.ticklabel_format(axis="x", style='sci', scilimits=(-2,-2))
ax.set_xlim([10*1e-2,19*1e-2])
plt.gca().invert_yaxis()
plt.show()
fig.savefig('barplots3_41.jpg', dpi=300, bbox_inches="tight")



df_64 = df_64.iloc[0:4, :]
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.5, 0.4])
ax.barh(df_64.iloc[:, 0], df_64.iloc[:, 1], height=0.7, color="#530B13")
ax.set_axisbelow(True)
ax.grid(True, color='k', linestyle='--', axis='x', linewidth=0.3, alpha=0.3)
ax.set_xlabel("Feature Importance ")
ax.ticklabel_format(axis="x", style='sci', scilimits=(-2,-2))
# ax.set_xlim([10*1e-2,19*1e-2])
plt.gca().invert_yaxis()
# fig = ax.get_figure()
plt.show()
fig.savefig('barplots4_41.jpg', dpi=300, bbox_inches="tight")


