# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:46:57 2021

@author: toonv
"""

def emulator_v2_graphs(cubic_output, root_output):
    
    reds = [i for i in range(len(cubic_output))]
    
    for red in zip(*root_output):
        plt.plot(reds, red, label='root')
    plt.title('accuracy of emulator')
    plt.xlabel('number of tested carbon tax paths')
    plt.ylabel('reduction')
    
    for red in zip(*cubic_output):
        plt.plot(reds, red, label='cubic')
    
    plt.legend()
    
    