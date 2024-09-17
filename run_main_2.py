import pyNN.nest as sim

import os
import gen_net
from net_check import *
from net_gen import *


random.seed(100)

sim_duration = 3000

# Connect spiking source to neurons
num_spike = 10  # number of spike sources
conn_spike = 1  # number of connections from spike to neuron
spike_connected = 10  # number of neurons connected by spike source
weight_spike = 0.5  # weight from spike to neuron, at 50Hz minimum is 0.05
delay_spike = 1.0  # unfied delays

# Connection between neurons
num_neurons = 1000
neuron_connected = 10  # number of neurons connected to each other
weight_neuron = 0.05  # weight from neuron to neuron, at 50Hz minimum is 0.05
delay_neuron = 1.0  # unfied delays

spike_conn_matrix = [[] for _ in range(num_spike)]
for i in range(num_spike):
    spike_conn_matrix[i].append([i, weight_spike, delay_neuron])
    if conn_spike > 1:
        targets = list(range(num_neurons))
        targets.remove(i)
        random.shuffle(targets)
        for _ in range(conn_spike - 1):
            spike_conn_matrix[i].append([targets.pop(0), weight_spike, delay_neuron])
spike_conn = matrix2connector(spike_conn_matrix)

spike_conn_matrix = [[]]
for i in random.sample(list(range(1000)), spike_connected):
    spike_conn_matrix[0].append([i, weight_spike, delay_spike])
spike_conn = matrix2connector(spike_conn_matrix)

spike_times = gen_net.spike_poisson(num_spike, conn_spike)


def run_random_fixin(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_source = sim.Population(num_spike, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_fixin(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_fixed output_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_fixed output_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_random_fixout(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_fixout(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_fixed output_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_fixed output_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_random_linear(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_fixout_linear(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_Linear_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_Linear_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_random_inverse(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_fixout_inverse(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_inverse_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_inverse_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_random_normal(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_fixout_normal(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_normal_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_normal_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_random_er(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_random_er(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    spike_save(spike_data, 'random_er_' + str(weight_neuron) + '.txt')
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    return binned_count
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Random_normal_ER_Weight: ' + str(weight_neuron))
    # plt.savefig('Random_normal_ER_Weight: ' + str(weight_neuron) + '.png')


def run_small_ws(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp(tau_refrac=2.0))

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_small_ws(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Small World WS - Weight: ' + str(weight_neuron))
    # plt.savefig('Small_WS_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_small_ws_rgr(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_small_ws_rgr(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.xlabel('Time(ms)')
    # plt.ylabel('Neurons Fired in 10 ms rolling window')
    # plt.title('Small World Watts–Strogatz Rich get Richer - Weight: ' + str(weight_neuron))
    # plt.savefig('Small_WS_RGR_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_small_ws_fit(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_small_ws_fit(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.xlabel('Time(ms)')
    # plt.ylabel('Neurons Fired in 10 ms rolling window')
    # plt.title('Small World Watts–Strogatz Rich get Richer - Weight: ' + str(weight_neuron))
    # plt.savefig('Small_WS_RGR_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_small_chris(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_small_chris(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Small World Chris - Weight: ' + str(weight_neuron))
    # plt.savefig('Small_small_Chris_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_scalefree_ba(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_scalefree_ba(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Scale Free_BA - Weight: ' + str(weight_neuron))
    # plt.savefig('Scale free_BA_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_scalefree_price(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_scalefree_price(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Scale Free_Price - Weight: ' + str(weight_neuron))
    # plt.savefig('Scale free_Price_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


def run_scalefree_ptc(weight_neuron=0.01):
    # Setup simulation parameters
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=10.0)

    # Create spiking source
    spike_times = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    # Create group of IF_cond_exp neurons
    neurons = sim.Population(num_neurons, sim.IF_cond_exp())

    conn_list = sim.FromListConnector(spike_conn)
    sim.Projection(spike_source, neurons, conn_list)
    [current_net, _] = conn_scalefree_ptc(num_neurons, neuron_connected)
    neuron_conn_matrix = net2matrix(current_net, weight_neuron, delay_neuron)
    conn_list = sim.FromListConnector(matrix2connector(neuron_conn_matrix))
    sim.Projection(neurons, neurons, conn_list)

    neurons.record({'spikes'})

    # Run simulation
    sim.run(sim_duration)
    # End simulation
    sim.end()

    spike_data = neurons.get_data().segments[0].spiketrains
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(num_neurons):
        spike_timings = spike_data[i]
        if len(spike_timings) > 1:
            for j in spike_timings:
                for k in range(-5, 5):
                    log_binned = math.ceil(float(j) + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    # plt.figure()
    # plt.plot(binned_count)
    # plt.axvline(x=280, color='b')
    # plt.xlim(0, sim_duration)
    # plt.suptitle('Scale Free_ptc - Weight: ' + str(weight_neuron))
    # plt.savefig('Scale free_ptc_Weight: ' + str(weight_neuron) + '.png')
    return binned_count


del_png()
random.seed(1000)
binned_matrix = []

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Time')
ax.set_ylabel('Weight')
ax.set_zlabel('Neuron Count')
from datetime import datetime

plt.minorticks_off()
count = 0
xline = range(sim_duration)

for i in range(60, 200, 5):
    random.seed(1000)
    binned_count = run_small_ws(i / 10000)
    yline = [i for _ in binned_count]
    ax.plot3D(xline, yline, binned_count)
    binned_matrix.append(binned_count)
    ax.set_title('Random_ws')

spike_save(binned_matrix, 'binned_routine.txt')
plt.savefig('simple_' + str(datetime.now()) + '.png')
plt.show()


duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
