import random
from datetime import datetime

import matplotlib.pyplot as plt
import pyNN.nest as sim

import gen_net

sim_duration = 2000
delay = 1.0  # milliseconds
refrac_period = 2.0  # refractory period for spike source and neurons

# spiking source to neurons
num_spike = 10
spike_count = 20  # number of spikes generated for each spike source
spike_interval = 10  # inter spike interval
conn_spike = 2  # number of connection from spike to neuron, >1: random connection
weight_spike = 0.5  # weight from spike to neurons

# neuron to neuron
num_neurons = 1000
conn_neuron = 10  # synaptic connection between neurons
weight_neuron = 0.05  # weight from neuron to neuron, at 50Hz minimum is 0.05

spike_conn_matrix = [[] for _ in range(num_spike)]
for i in range(num_spike):
    spike_conn_matrix[i].append([i, weight_spike, delay])
    if conn_spike > 1:
        targets = list(range(num_neurons))
        targets.remove(i)
        random.shuffle(targets)
        for _ in range(conn_spike - 1):
            spike_conn_matrix[i].append([targets.pop(0), weight_spike, delay])
spike_conn = gen_net.matrix2connector(spike_conn_matrix)

spike_train = gen_net.spike_grid()


def run_once(weight_neuron=0.01, conn_matrix=[], spike_train=[]):
    sim.setup(timestep=0.1)

    # Create spiking source
    spike_source = sim.Population(num_spike, sim.SpikeSourceArray(spike_times=spike_train))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp(tau_refrac=refrac_period))

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    neuron_conn_matrix = gen_net.net2matrix(conn_matrix, weight_neuron, delay)
    conn_list = sim.FromListConnector(gen_net.matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()
spike_train
    spike_data = neurons.get_data().segments[0].spiketrains
    gen_net.spike_save(spike_data)
    return gen_net.spike_binned(spike_data=spike_data, sim_duration=sim_duration)


plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Time')
ax.set_ylabel('Weight')
ax.set_zlabel('Neuron Count')

random.seed(1000)
conn_matrix = gen_net.conn_random()
spike_train = gen_net.spike_poisson()

a = 50
b = 500
for i in range(a, b, 50):
    spike_binned_data = run_once(weight_neuron=float(i) / 100, conn_matrix=conn_matrix, spike_train=spike_train)
    xline = range(sim_duration)
    yline = [i for _ in xline]
    zline = spike_binned_data
    ax.plot3D(xline, yline, zline)
plt.savefig('sample' + str(datetime.now()) + '.png')
plt.show()

print('Done')
