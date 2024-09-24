import os
import random
from datetime import datetime

import matplotlib.pyplot as plt

import gen_net

time_delay = 1.0  # milliseconds
time_refrac = 1.0  # refractory period

num_spike_source = 10  # number of spike source
conn_spike_neuron = 1  # connection
spike_count = 20  # number of spikes
spike_interval = 10  # inter spike interval
weight_spike_neuron = 0.5  # weight from spike to neuron, at 50Hz minimum is 0.05

num_neuron = 1000
conn_neuron_neuron = 10  # number of connection between neurons
weight_neuron_neuron = 0.001  # weight between neurons

sim_duration = 3000
weight_range = range(70, 90)

for _ in range(1):

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Neuron Count')

#    random_seed = int(random.random() * 10000)
    random_seed = 4838
    random.seed(random_seed)

    for i in weight_range:
        spike_train = gen_net.spike_train_poisson(num_spike_source, spike_count, spike_interval,
                                                  time_refrac)
        conn_net_spike = gen_net.conn_spike_neuron(num_spike_source, num_neuron, conn_spike_neuron)
        conn_matrix_spike = gen_net.net2matrix(conn_net_spike, weight_spike_neuron)
        conn_net_neuron = gen_net.conn_r_normal(num_neuron, conn_neuron_neuron)
        conn_matrix_neuron = gen_net.net2matrix(conn_net_neuron, float(i) / 10000)
        [spike_binned_unique, spike_data] = gen_net.run_spike_neuron(conn_matrix_spike, conn_matrix_neuron, spike_train,
                                                                     sim_duration)

        xline = range(sim_duration)
        yline = [i for _ in xline]
        zline = spike_binned_unique
        ax.plot3D(xline, yline, zline)

    plt.savefig('Plots/' + str(datetime.now()) + '_Seed_' + str(random_seed) + '.png')

    os.system('play -nq -t alsa synth {} sine {}'.format(2, 440))  # 2 second at 440hz
