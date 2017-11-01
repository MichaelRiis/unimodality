import numpy as np
import pylab as plt
import seaborn as snb
from os.path import isfile, join


import sys
sys.path.append('../../code')
from util import plot_with_uncertainty2



#############################################################################################################
# Load and plot 
#############################################################################################################
Ns = [4, 6]

colors = snb.color_palette()
names = ['Regular', 'Regular + mean function', 'Unimodal']


for N in Ns:

    # load data
    raw_data = np.load('densityN%d.npz' % N)

    lmp = raw_data['lmp']
    betas = raw_data['betas']


    # plot    
    snb.set(font_scale=1.5)
    fig = plt.figure()


    for idx, (name, color) in enumerate(zip(names, colors)):

        data = raw_data[name][()]
        mean, lower, upper = data['mean'], data['lower'], data['upper']

        if name == "Regular + mean function":
            style = '--'
        else:
            style='-'

        plot_with_uncertainty2(betas, mean, lower=lower, upper=upper, label=name, color=color, linestyle=style)
        plt.grid(True)

    plt.plot(betas, np.exp(lmp), 'k--', linewidth=3, label='Target', alpha=0.5)

    plt.grid(True)
    plt.legend()
    plt.xlabel('$\\beta$')
    plt.ylabel('Unnormalized density')

    path = '../../paper/figures/'
    # fig.savefig(path + 'density_estimatesN%d.png' % N, bbox_inches=None)

    plt.show(block = False)
