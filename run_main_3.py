import os
from datetime import datetime
import random

import matplotlib.pyplot as plt

import gen_plot

data_folder = 'Data'
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

current_folder = data_folder + '/' + str(datetime.now())
if not os.path.isdir(current_folder + '/Plots'):
    os.makedirs(current_folder + '/Plots')
if not os.path.isdir(current_folder + '/Spikes'):
os.makedirs(current_folder + '/Spikes')

conn_file = data_folder + '/conn.txt'
weight_file = data_folder + '/weight.txt'
spike_source_file = data_folder + '/spike_source.txt'

