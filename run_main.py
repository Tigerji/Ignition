import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import pyNN.nest as sim

import gen_net
import gen_plot
from gen_plot import plot_folder

sim_duration = 3000

time_delay = 1.0  # milliseconds
time_refrac = 1.0  # refractory period

num_spike_source = 10  # number of spike source
conn_spike = 1 # connection
spike_count = 20  # number of spikes
spike_interval = 10  # inter spike interval
weight_spike_neuron = 0.5  # weight

num_neuron = 1000
conn_neuron = 10
weight_neuron_neuron = 0.05

random.seed(1000)

spike_train = gen_net.spike_train_square(num_spike_source, spike_count, spike_interval,
                                         time_refrac)  # num_spike_source=10, spike_count=20, spike_interval=10, refractory=1, start_time=100


def run_random(weight_neuron=0.01):
    conn_matrix = gen_net.conn_neuron_random(num_neuron, conn_neuron)

    sim.setup(timestep=0.1)

    spike_source = sim.Population(num_spike_source, sim.SpikeSourceArray(spike_times=spike_train))
    neurons = sim.Population(num_neuron, sim.IF_cond_exp(tau_refrac=time_refrac))

    neuron_conn_matrix = gen_net.net2matrix(conn_matrix, weight_neuron, time_delay)
    conn_list = sim.FromListConnector(gen_net.matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    conn_matrix = gen_net.conn_spike_neuron(num_spike_source, num_neuron, conn_spike)
    neuron_conn_matrix = gen_net.net2matrix(conn_matrix, weight_spike_neuron, time_delay)
    conn_list = sim.FromListConnector(gen_net.matrix2connector(neuron_conn_matrix))
    sim.Projection(spike_source, neurons, conn_list)

    neurons.record({'spikes'})
    sim.run(sim_duration)
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    gen_net.spike_save(spike_data)
    return gen_net.spike_binned(spike_data, sim_duration)


plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Time')
ax.set_ylabel('Weight')
ax.set_zlabel('Neuron Count')

weight_range = [10,20]
weight_step = 1

for i in range(weight_range[0], weight_range[1], weight_step):
    spike_binned_data = run_random(weight_neuron=float(i) / 1000)
    xline = range(sim_duration)
    yline = [i for _ in xline]
    zline = spike_binned_data
    ax.plot3D(xline, yline, zline)
plt.savefig('Plots/sample' + str(datetime.now()) + '.png')



# random.seed(1000)
# conn_matrix = gen_net.conn_random()
# spike_train = gen_net.spike_poisson()
#
# a = 50
# b = 500
# for i in range(a, b, 50):
#     spike_binned_data = run_once(weight_neuron=float(i) / 100, conn_matrix=conn_matrix, spike_train=spike_train)
#     xline = range(sim_duration)
#     yline = [i for _ in xline]
#     zline = spike_binned_data
#     ax.plot3D(xline, yline, zline)
# plt.savefig('sample' + str(datetime.now()) + '.png')
# plt.show()
#
# print('Done')

def plot_spike_demo():
    plt.figure()
    plt.subplot(3, 1, 1)
    spike_times = gen_net.spike_train_grid(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Grid')

    plt.subplot(3, 1, 2)
    spike_times = gen_net.spike_train_poisson(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Poisson')

    plt.subplot(3, 1, 3)
    spike_times = gen_net.spike_train_random(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Random')
    plt.tight_layout()

    plt.figure()
    plt.subplot(3, 1, 1)
    spike_times = gen_net.spike_train_triangular(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Triangular')

    plt.subplot(3, 1, 2)
    spike_times = gen_net.spike_train_square(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Square')

    plt.subplot(3, 1, 3)
    spike_times = gen_net.spike_train_triangular(num_spike_source, spike_count, spike_interval, time_refrac)
    gen_plot.plot_spike_train(spike_times)
    plt.title('Normal')
    plt.tight_layout()
    plt.show()


os.system('play -nq -t alsa synth {} sine {}'.format(2, 440))  # 2 second at 440hz
plt.show()