# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:28:41 2021

@author: mnakhaei
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
os.chdir('C:\Mostafa\Python\Parisa')


df_mancus = pd.read_excel("marcellus_IOU.xlsx")
mpl.rcParams.update(mpl.rcParamsDefault)
#or
plt.style.use('default')

def print_it(style="fast", save_name1="X_PNG.png", save_name2="y_TIF.tif"):
### https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py


    plt.rcParams.update({'font.size': 10})
    plt.style.use(style)
    
    print(plt.style.available) 
    font = {'family': 'san-serif',
            'color':  'k',
            'weight': 'normal',
            'size': 10,
            }
    
    fig, ax = plt.subplots(2,2)
    x1 = ["'0'", "'1'", "'2'", "'3'", "'4'"]
    x_pos = np.arange(len(x1)) 
    width = 0.17
    h1 = "////"
    h2 = "+++"
    
    ax[0,0].set_xlabel('Class')
    ax[0,1].set_xlabel('Class')
    ax[1,0].set_xlabel('Class')
    ax[1,1].set_xlabel('Class')
    
    
    # mpl.rcParams["hatch.color"] =  "r"
    ##################################
    ########### subplot 1#############
    ##################################
    y3 = [df_mancus[0][0], df_mancus[1][0], df_mancus[2][0], df_mancus[3][0], df_mancus[4][0]]
    y4 = [df_mancus[0][1], df_mancus[1][1], df_mancus[2][1], df_mancus[3][1], df_mancus[4][1]]
    y5 = [df_mancus[0][2], df_mancus[1][2], df_mancus[2][2], df_mancus[3][2], df_mancus[4][2]]
    ax[0, 0].grid(axis='y', color='k', linestyle='--', linewidth=0.2, alpha=0.3, zorder=1)
    G1 = ax[0, 0].bar(x_pos - width, y3, width, label='OI', hatch=h1)
    G2 = ax[0,0].bar(x_pos, y4, width, label='14 FT')
    G3 = ax[0,0].bar(x_pos + width, y5, width, label='VGG16', hatch=h2)

    
    ax[0,0].set_xticks(x_pos,) #
    ax[0,0].set_xticklabels(x1, fontdict=font)
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_ylabel('Score')
    ax[0,0].set_ylim([0, 1.1])

    ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.set_title('Scores by group and gender')
    # ax[0,0].legend(loc="upper left", bbox_to_anchor=(0.6,1.5))
    ax[0,0].legend(bbox_to_anchor=(0.2, 1.07, 2, .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., edgecolor='k')
    
    textstr = "RF"
    props = dict(boxstyle='round', facecolor='none', alpha=0.5)
    ax[0,0].text(0.05, 0.95, textstr, transform=ax[0,0].transAxes,
                 fontsize=11, verticalalignment='top', bbox=props)

    
    ##################################
    ########### subplot 2#############
    ##################################
    y3 = [df_mancus[0][3], df_mancus[1][3], df_mancus[2][3], df_mancus[3][3], df_mancus[4][3]]
    y4 = [df_mancus[0][4], df_mancus[1][4], df_mancus[2][4], df_mancus[3][4], df_mancus[4][4]]
    y5 = [df_mancus[0][5], df_mancus[1][5], df_mancus[2][5], df_mancus[3][5], df_mancus[4][5]]
    ax[0, 1].grid(axis='y', color='k', linestyle='--', linewidth=0.2, alpha=0.3, zorder=1)

    G1 = ax[0, 1].bar(x_pos - width, y3, width, label='OI', hatch=h1)
    G2 = ax[0,1].bar(x_pos, y4, width, label='14 FT')
    G3 = ax[0,1].bar(x_pos + width, y5, width, label='VGG16', hatch=h2)
    ax[0,1].set_xticks(x_pos)
    ax[0,1].set_xticklabels(x1, fontdict=font)
    ax[0,1].set_ylim([0,1])
    ax[0,1].set_ylabel('Score')
    ax[0,1].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[0,1].set_ylim([0, 1.1])
    textstr = "FNN"
    props = dict(boxstyle='round', facecolor='none', alpha=0.5)
    ax[0,1].text(0.05, 0.95, textstr, transform=ax[0,1].transAxes,
                 fontsize=11, verticalalignment='top', bbox=props)
    ##################################
    ########### subplot 3#############
    ##################################
    y3 = [df_mancus[0][6], df_mancus[1][6], df_mancus[2][6], df_mancus[3][6], df_mancus[4][6]]
    y4 = [df_mancus[0][7], df_mancus[1][7], df_mancus[2][7], df_mancus[3][7], df_mancus[4][7]]
    y5 = [df_mancus[0][8], df_mancus[1][8], df_mancus[2][8], df_mancus[3][8], df_mancus[4][8]]
    ax[1, 0].grid(axis='y', color='k', linestyle='--', linewidth=0.2, alpha=0.3, zorder=1)

    G1 = ax[1, 0].bar(x_pos - width, y3, width, label='OI', hatch=h1)
    G2 = ax[1,0].bar(x_pos, y4, width, label='14 FT')
    G3 = ax[1,0].bar(x_pos + width, y5, width, label='VGG16', hatch=h2)
    ax[1, 0].set_xticks(x_pos)
    ax[1, 0].set_xticklabels(x1, fontdict=font)
    ax[1, 0].set_ylim([0,1])
    ax[1, 0].set_ylabel('Score')
    ax[1,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[1,0].set_ylim([0, 1.1])
    textstr = "K-Means"
    props = dict(boxstyle='round', facecolor='none', alpha=0.5)
    ax[1,0].text(0.05, 0.95, textstr, transform=ax[1,0].transAxes,
                 fontsize=11, verticalalignment='top', bbox=props)
    ##################################
    ########### subplot 4#############
    ##################################
    y3 = [df_mancus[0][9], df_mancus[1][9], df_mancus[2][9], df_mancus[3][9], df_mancus[4][9]]
    y4 = [df_mancus[0][10], df_mancus[1][10], df_mancus[2][10], df_mancus[3][10], df_mancus[4][9]]
    y5 = [df_mancus[0][11], df_mancus[1][11], df_mancus[2][11], df_mancus[3][11], df_mancus[4][9]]
    ax[1, 1].grid(axis='y', color='k', linestyle='--', linewidth=0.2, alpha=0.3, zorder=1)

    G1 = ax[1, 1].bar(x_pos - width, y3, width, label='OI', hatch=h1)
    G2 = ax[1,1].bar(x_pos, y4, width, label='14 FT')
    G3 = ax[1,1].bar(x_pos + width, y5, width, label='VGG16', hatch=h2)
    ax[1,1].set_xticks(x_pos)
    ax[1,1].set_xticklabels(x1, fontdict=font)
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_ylabel('Score')
    ax[1,1].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[1,1].set_ylim([0, 1.1])
    textstr = "U-Net"
    props = dict(boxstyle='round', facecolor='none', alpha=0.5)
    ax[1,1].text(0.05, 0.95, textstr, transform=ax[1,1].transAxes,
                 fontsize=11, verticalalignment='top', bbox=props)
    plt.rc('axes', axisbelow=True)

    fig.tight_layout()
    plt.show()
    
    
    fig.savefig(save_name1, dpi=600)
    fig.savefig(save_name2, dpi=600)


print_it(style='fast',
         save_name1="Marcellus_IOUPNG.png",
         save_name2="Marcellus_IOUTIF.tif")