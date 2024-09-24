import matplotlib.pyplot as plt
import pyNN.nest as sim

num_spike = 10
num_neuron = 20

sim.setup(timestep=0.1)
spike_train = []
for spike_timing in range(100, 200, 10):
    spike_train.append(spike_timing)

spike_source = sim.Population(num_spike, sim.SpikeSourceArray(spike_times=spike_train))
neurons = sim.Population(num_neuron, sim.IF_cond_exp(tau_refrac=1.0))
neurons.record({'spikes'})
syn = sim.StaticSynapse(weight=0.5, delay=1.0)
sim.Projection(spike_source, neurons, sim.AllToAllConnector(), syn)

sim.run(1000)
sim.end()

spike_data = neurons.get_data().segments[0].spiketrains
spike_data_float = [[] for _ in range(num_neuron)]
for current_neuron in range(num_neuron):
    if len(spike_data[current_neuron]) > 0:
        for current_spike in spike_data[current_neuron]:
            spike_data_float[current_neuron].append(float(current_spike))

plt.figure()
for i in range(num_neuron):
    if spike_data_float[i]:
        plt.plot(spike_data_float[i], [i for _ in spike_data_float[i]], 'b,')
plt.xlabel('Time')
plt.ylabel('Neurons')
plt.show()
