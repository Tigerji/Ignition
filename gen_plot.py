import math
from datetime import datetime

import matplotlib.pyplot as plt

import gen_net

plot_folder = 'Plots'


def plot_spike_train(spike_train, x_range=None):
    if x_range is None:
        x_range = [50, 350]
    neuron_count = 1
    spike_count = [0 for _ in range(0, x_range[1])]
    for i in spike_train:
        for j in i:
            plt.plot(j, neuron_count, 'b.')
            for s in range(j - 5, j + 5):
                if 0 < s < x_range[1]:
                    spike_count[s] = spike_count[s] + 1
        neuron_count = neuron_count + 1
    plt.plot(spike_count, 'r-')
    plt.xlabel("Spike Timing")
    plt.ylabel("Neurons")
    plt.xlim(x_range)
#    plt.savefig(plot_folder + '/' + plot_name + '_' + str(datetime.now()) + '.png')


def plot_conn_net(conn_net, title='Connectivity Map', plot_name='conn_net'):
    plt.figure()
    num_neuron = len(conn_net)
    for i in range(num_neuron):
        if conn_net[i]:
            for j in conn_net[i]:
                plt.plot(i, j, 'b.')
    plt.xlim(0, num_neuron)
    plt.ylim(0, num_neuron)
    plt.xlabel('Pre-synaptic neuron')
    plt.ylabel('Target neuron')
    plt.title(title)
    plt.savefig(plot_folder + '/' + plot_name + '_' + str(datetime.now()) + '.png')


def plot_binned(spike_data, sim_duration=1000, plot_name='spike_binned'):
    # plot spike data in binned format
    binned_fired = [[] for _ in range(sim_duration)]
    for i in range(len(spike_data)):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    a = plt.figure()
    plt.plot(binned_count)
    plt.xlim(0, sim_duration)
    plt.savefig(plot_folder + '/' + plot_name + '_' + str(datetime.now()) + '.png')
    plt.close(a)
    return binned_count


def plot3d(spike_binned_data, weight_range=[10, 20], sim_duration=1000, plot_name='spike_3d'):
    plt.minorticks_off()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Neuron Count')
    xline = range(sim_duration)
    count = 0
    for i in range(weight_range[0], weight_range[1]):
        yline = [i for _ in xline]
        zline = spike_binned_data
        count = count + 1
    ax.plot3D(xline, yline, zline)


#    plt.savefig(plot_folder + '/' + plot_name + '_' + str(datetime.now()) + '.png')


def plot_connectivity(spike_conn):
    a = plt.figure()
    for i in range(len(spike_conn)):
        if len(spike_conn[i]) > 0:
            for j in spike_conn[i]:
                plt.plot(i, j, 'b,')
    plt.title('Connectivity Map')
    plt.xlabel('Pre-Synaptic Neuron')
    plt.ylabel('Target Neurons')
    plt.savefig((plot_folder + '/connectivity_' + str(datetime.now()) + '.png'))
    plt.close(a)


def plot_spike_demo(num_spike_source=10, spike_count=20, spike_interval=10, time_refrac=1.0):
    plt.figure()
    plt.subplot(3, 1, 1)
    spike_times = spike_train_grid(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Grid')

    plt.subplot(3, 1, 2)
    spike_times = gen_net.spike_train_poisson(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Poisson')

    plt.subplot(3, 1, 3)
    spike_times = gen_net.spike_train_random(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Random')
    plt.tight_layout()

    plt.figure()
    plt.subplot(3, 1, 1)
    spike_times = gen_net.spike_train_triangular(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Triangular')

    plt.subplot(3, 1, 2)
    spike_times = gen_net.spike_train_square(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Square')

    plt.subplot(3, 1, 3)
    spike_times = gen_net.spike_train_triangular(num_spike_source, spike_count, spike_interval, time_refrac)
    plot_spike_train(spike_times)
    plt.title('Normal')
    plt.tight_layout()
    plt.show()
