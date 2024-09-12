import math
from datetime import datetime

import matplotlib.pyplot as plt

plotfolder = 'Plots/'


def plot_spike_train(spike_train, plotname='spiketrain'):
    plt.figure()
    neuron_count = 1
    for i in spike_train:
        print(i)
        for j in i:
            print(j, neuron_count)
            plt.plot(j, neuron_count, 'b.')
        neuron_count = neuron_count + 1
    plt.xlabel("Spike Timing")
    plt.ylabel("Neurons")
    plt.title(str(len(spike_train)) +
              ' spiking sources with ' +
              str(len(spike_train[0])) + ' spikes')
    plt.savefig(plotfolder + '/' + plotname + '_' + str(datetime.now()) + '.png')
    plt.show()


def plot_conn_net(conn_net, title='demo', plotname='conn_net'):
    num_neuron = len(conn_net)
    for i in range(num_neuron):
        if conn_net[i]:
            x = [i for j in conn_net[i]]
            plt.plot(x, conn_net[i], 'b,')
    plt.xlim(0, num_neuron)
    plt.ylim(0, num_neuron)
    plt.title(title)
    plt.savefig(plotfolder + '/' + plotname + '_' + str(datetime.now()) + '.png')


def plot_binned(spike_data, sim_duration=1000, plotname='spike_binned'):
    # plot spike data in binned format
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(len(spike_data)):
        if spike_data[i]:
            for j in spike_data[i]:
                for k in range(-5, 5):
                    log_binned = math.ceil(j + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    plt.plot(binned_count)
    plt.xlim(0, sim_duration)
    plt.savefig(plotfolder + '/' + plotname + '_' + str(datetime.now()) + '.png')

