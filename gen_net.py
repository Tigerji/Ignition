import math
import random
from datetime import datetime

import numpy


def spike_grid(num_source=10, num_spike=20, spike_interval=10, start_time=100, refractory=2):
    # grid spike train
    one_spike = []
    for i in range(num_spike):
        if spike_interval < refractory:
            one_spike.append(one_train.append(start_time + i * refractory))
        else:
            one_spike.append(start_time + i * spike_interval)
    return [one_spike for _ in range(num_source)]


def spike_poisson(num_source=10, num_spike=20, spike_interval=10, refractory=2, start_time=100):
    # each spike train is a poisson process
    spike_train = []
    for _ in range(num_source):
        poisson_count = numpy.random.poisson(lam=spike_interval, size=num_spike)
        one_spike = []
        spike_time = start_time
        for i in poisson_count:
            if i < refractory:
                spike_time = spike_time + refractory
            else:
                spike_time = spike_time + i
            one_spike.append(spike_time)
        spike_train.append(one_spike)
    return spike_train


def spike_random(num_source=10, num_spike=20, spike_interval=10, refractory=2, start_time=100):
    # spikes are random located in 2*spike_interval
    spike_train = []
    for _ in range(num_source):
        one_spike = []
        current_time = start_time + random.randrange(spike_interval * 2)
        one_spike.append(current_time)
        for j in range(num_spike - 1):
            current_time = current_time + random.randrange(refractory, spike_interval * 2)
            one_spike.append(current_time)
        spike_train.append(one_spike)
    return spike_train


def spike_triangular(num_source=10, num_spike=20, spike_interval=10, refractory=2, start_time=100):
    # weight follows triangular linear distribution
    spike_train = []
    peak_time = start_time + (num_spike - 1) * spike_interval / 2
    spike_time = list(range(start_time, start_time + (num_spike - 1) * spike_interval + 1, refractory))
    spike_weight_count = [peak_time - start_time - abs(i - peak_time) for i in spike_time]
    spike_weight = [i / sum(spike_weight_count) for i in spike_weight_count]
    for _ in range(num_source):
        one_spike = numpy.random.choice(spike_time, size=num_spike, replace=False, p=spike_weight)
        spike_train.append(sorted(one_spike))
    return spike_train


def spike_t_square(num_source=10, num_spike=20, spike_interval=10, refractory=2, start_time=100):
    # weight follows squares of triangular distribution
    spike_train = []
    peak_time = start_time + (num_spike - 1) * spike_interval / 2
    spike_time = list(range(start_time, start_time + (num_spike - 1) * spike_interval + 1, refractory))
    spike_weight_count = [(peak_time - start_time - abs(i - peak_time)) ** 2 for i in spike_time]
    spike_weight = [i / sum(spike_weight_count) for i in spike_weight_count]
    for _ in range(num_source):
        one_spike = numpy.random.choice(spike_time, size=num_spike, replace=False, p=spike_weight)
        spike_train.append(sorted(one_spike))
    return spike_train


def spike_t_quadrad(num_source=10, num_spike=20, spike_interval=10, refractory=2, start_time=100):
    # weight follows quadrads of triangular distribution
    spike_train = []
    peak_time = start_time + (num_spike - 1) * spike_interval / 2
    spike_time = list(range(start_time, start_time + (num_spike - 1) * spike_interval + 1, refractory))
    spike_weight_count = [(peak_time - start_time - abs(i - peak_time)) ** 4 for i in spike_time]
    spike_weight = [i / sum(spike_weight_count) for i in spike_weight_count]
    for _ in range(num_source):
        one_spike = numpy.random.choice(spike_time, size=num_spike, replace=False, p=spike_weight)
        spike_train.append(sorted(one_spike))
    return spike_train


def conn_random(num_neuron=10, num_conn=4):
    # random network
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        targets.remove(i)
        selected = random.sample(targets, num_conn)
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net


def conn_r_linear(num_neuron=10, num_conn=4):
    # weight is linearly distributed based on distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 if i == j else num_neuron / 2 - min(abs(i - j), num_neuron - abs(i - j)) for j in targets]
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net


def conn_r_inverse(num_neuron=10, num_conn=4):
    # random network where weight is the inverse of distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 if j == i else 1 / min(abs(i - j), num_neuron - abs(i - j)) for j in targets]
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net


def conn_r_normal(num_neuron=10, num_conn=4, sigma=20):
    # random network where weight is normal distributed based on distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 for _ in range(num_neuron)]
        for j in targets:
            if i != j:
                s = min(abs(i - j), num_neuron - abs(i - j)) / num_neuron * sigma
                weight[j] = 1 / (1 * math.sqrt(2 * math.pi)) * math.exp(-((s - 0) ** 2 / (2 * 1 ** 2)))
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net


def conn_r_er(num_neuron=10, num_conn=4):
    # random network with Erdős–Rényi selection on edges
    conn_prob = num_conn / num_neuron
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        for j in range(num_neuron):
            if random.random() < conn_prob and i != j:
                conn_net[i].append(j)
    return conn_net


def conn_small_ws(num_neuron=10, num_conn=4, mut_ratio=0.05):
    # small world network with Watts-Strogatz model
    conn_net = [[] for _ in range(num_neuron)]
    stats_net = [num_conn for _ in range(num_neuron)]
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws')
    else:
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)
            for j in range(num_conn):  # mutation
                mut_target = list(range(num_neuron))
                del mut_target[i]  # no self connection
                for k in curr_net:
                    mut_target.remove(k)  # no repetition
                if random.random() < mut_ratio:
                    new_target = random.sample(mut_target, 1)
                    stats_net[curr_net[j]] = stats_net[curr_net[j]] - 1
                    curr_net[j] = new_target[0]
                    stats_net[curr_net[j]] = stats_net[curr_net[j]] + 1
            conn_net[i] = sorted(curr_net)
    return conn_net


def conn_small_ws_rgr(num_neuron=10, num_conn=4, mut_ratio=0.05):
    # small world model with Watts-Strogatz but richer get richer
    conn_net = [[] for i in range(num_neuron)]
    stats_net = [num_conn for i in range(num_neuron)]
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws_rgr')
    else:
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)
            conn_net[i].extend(curr_net)

        # mutation
        shuffled_neuron = list(range(num_neuron))
        random.shuffle(shuffled_neuron)
        shuffled_conn = list(range(num_conn))
        random.shuffle(shuffled_conn)
        for j in shuffled_conn:
            for i in shuffled_neuron:
                mut_target = list(range(num_neuron))
                temp_stats = stats_net[:]
                temp_stats[i] = 0  # no self connection
                for k in conn_net[i]:
                    temp_stats[k] = 0  # no repetition
                if random.random() < mut_ratio:
                    new_target = random.choices(mut_target, temp_stats)  # rich get richer
                    stats_net[conn_net[i][j]] = stats_net[conn_net[i][j]] - 1
                    conn_net[i][j] = new_target[0]
                    stats_net[conn_net[i][j]] = stats_net[conn_net[i][j]] + 1
        return conn_net


def conn_small_chris(num_neuron=10, num_conn=4, n_inout=1):
    # Chris model, modified Newman–Watts model
    if num_conn < n_inout or num_conn < 2:
        print('Parameter error in conn_small_chris')
    else:
        conn_net = [[] for i in range(num_neuron)]
        stats_net = [n_inout for i in range(num_neuron)]
        neuron_list = list(range(num_neuron))
        shuffled_list = neuron_list[:]
        for i in range(n_inout):
            count = 0
            while not count:
                random.shuffle(shuffled_list)
                for j, k in zip(neuron_list, shuffled_list):
                    if j == k or k in conn_net[j]:
                        break;
                else:
                    for j in range(num_neuron):
                        conn_net[j].append(shuffled_list[j])
                    count = count + 1
        for j in range(n_inout, num_conn):
            random.shuffle(shuffled_list)
            for i in shuffled_list:
                temp_stats = stats_net[:]
                temp_stats[i] = 0
                for k in conn_net[i]:
                    temp_stats[k] = 0
                new_target = random.choices(neuron_list, temp_stats)
                conn_net[i].append(new_target[0])
                stats_net[conn_net[i][j]] = stats_net[conn_net[i][j]] + 1

        return conn_net


def conn_scalefree_ba(num_neuron=10, num_conn=4):
    # Scale free network using Barabási–Albert model
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ba')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net = [0 for _ in range(num_neuron)]
        for i in range(num_conn + 1):
            neuron_list = list(range(num_conn + 1))
            neuron_list.remove(i)
            conn_net[i].extend(neuron_list)
            stats_net[i] = stats_net[i] + num_conn
        for i in range(num_conn + 1, num_neuron):
            neuron_list = list(range(i))
            temp_stats = stats_net[0:i]
            stats_net[i] = stats_net[i] + 1
            for j in range(num_conn):
                new_target = random.choices(neuron_list, temp_stats)
                conn_net[i].append(new_target[0])
                conn_net[new_target[0]].append(i)
                stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                stats_net[i] = stats_net[i] + 1
                temp_stats[new_target[0]] = 0
        return conn_net


def conn_scalefree_price(num_neuron=10, num_conn=4):
    # Scale free network Price model, directed BA
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ba')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net = [0 for _ in range(num_neuron)]
        for i in range(num_conn + 1):
            neuron_list = list(range(num_conn + 1))
            neuron_list.remove(i)
            conn_net[i].extend(neuron_list)
            stats_net[i] = stats_net[i] + num_conn
        for i in range(num_conn + 1, num_neuron):
            neuron_list = list(range(i))
            temp_stats = stats_net[0:i]
            stats_net[i] = stats_net[i] + 1
            for j in range(num_conn):
                new_target = random.choices(neuron_list, temp_stats)
                conn_net[i].append(new_target[0])
                stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                temp_stats[new_target[0]] = 0
        return conn_net


def conn_scalefree_ptc(num_neuron=10, num_conn=4, ratio=0.5, n_inout=1):
    # Scale free network modified from Pi, X., Tang, L., & Chen, X. (2021)
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ptc')
    else:
        conn_net = [[] for i in range(num_neuron)]
        stats_net_in = [0 for i in range(num_neuron)]
        stats_net_out = [0 for i in range(num_neuron)]
        neuron_list = list(range(num_neuron))
        shuffled_list = neuron_list[:]
        for i in range(n_inout):
            count = 0
            while not count:
                random.shuffle(shuffled_list)
                for j, k in zip(neuron_list, shuffled_list):
                    if j == k or k in conn_net[j]:
                        break;
                else:
                    for j in range(num_neuron):
                        conn_net[j].append(shuffled_list[j])
                        stats_net_in[j] = stats_net_in[j] + 1
                        stats_net_out[j] = stats_net_out[j] + 1
                    count = count + 1

        for i in range(num_conn + 1, num_neuron):
            neuron_list = list(range(i + 1))
            temp_stats_in = stats_net_in[0:(i + 1)]
            temp_stats_out = stats_net_out[0:(i + 1)]

            for _ in range(int(num_conn * ratio)):  # edges from new neuron
                new_target = random.choices(neuron_list, temp_stats_in)
                conn_net[i].append(new_target[0])
                stats_net_in[new_target[0]] = stats_net_in[new_target[0]] + 1
                temp_stats_in[new_target[0]] = 0
            stats_net_in[i] = stats_net_in[i] + 1
            temp_stats_in[i] = stats_net_in[i] + 1

            for _ in range(num_conn - int(num_conn * ratio)):  # edges from new neuron
                found = False  # from existing neurons
                while not found:
                    new_source = random.choices(neuron_list, temp_stats_out)
                    new_target = random.choices(neuron_list, temp_stats_in)
                    temp_stats_in[new_target[0]] = 0
                    temp_stats_out[new_source[0]] = 0

                    if new_target[0] not in conn_net[new_source[0]]:
                        conn_net[new_source[0]].append(new_target[0])
                        stats_net_in[new_target[0]] = stats_net_in[new_target[0]] + 1
                        stats_net_out[new_source[0]] = stats_net_out[new_source[0]] + 1
                        found = True
            stats_net_out[i] = stats_net_out[i] + int(num_conn * ratio)
        return conn_net


def net2matrix(conn_net, weight=0.01, delay=0.1, modifier=0):
    # conn_net to conn_matrix with fixed weights
    conn_matrix = [[] for _ in range(len(conn_net))]
    for i in range(len(conn_net)):
        for j in conn_net[i]:
            conn_matrix[i].append([j + modifier, weight, delay])
    return conn_matrix


def matrix2net(conn_matrix):
    # conn_matrix to conn_net
    conn_net = [[] for _ in range(len(conn_matrix))]
    weight_net = [[] for _ in range(len(conn_matrix))]
    delay_net = [[] for _ in range(len(conn_matrix))]
    for i in range(len(conn_matrix)):
        if conn_matrix[i]:
            for j in conn_matrix[i]:
                conn_net[i].append(j[0])
                weight_net[i].append(j[1])
                delay_net[i].append(j[2])
    return conn_net, weight_net, delay_net


def matrix2connector(conn_matrix, modifier=0):
    # conn_matrix to connector list
    connector = []
    for i in range(len(conn_matrix)):
        for j in conn_matrix[i]:
            if j:
                connector.append((i + modifier, j[0], j[1], j[2]))
    return connector


def connector2matrix(connector, num_neuron=10):
    # connector list to conn_matrix
    conn_matrix = [[] for _ in range(num_neuron)]
    for i in connector:
        conn_matrix[i[0]].append([i[1], i[2], i[3]])
    return conn_matrix


def matrix_save(matrix, data_file="matrixsave"):
    with open((data_file + str(datetime.now()) + 'txt'), 'w+') as f:
        for i in matrix:
            for j in i:
                f.write(str(j))
            f.write('\n')


def spike_save(spikes, data_file='spikesave'):
    with open((data_file + str(datetime.now()) + 'txt'), 'w+') as f:
        for i in range(len(spikes)):
            spike_number = []
            spike_timings = spikes[i]
            if len(spike_timings) > 1:
                for j in spike_timings:
                    spike_number.append("%.1f" % float(j))
            f.write(str(spike_number) + '\n')


def matrix_load(data_file='matrixfile.txt'):
    # load in matrix form from txt
    with open(data_file, 'r') as f:
        data = []
        for i in f.readlines():
            removed = i[1:-2].replace('[', '')
            temp_matrix = []
            if len(removed) > 0:
                sub_matrix = removed.split(']')
                for j in sub_matrix:
                    temp_matrix.append([int(j.split(',')[0]), float(j.split(',')[1])])
            data.append(temp_matrix)
    return data


def spike_load(data_file='spikefile.txt'):
    # load spike train data from txt
    with open(data_file, 'r') as f:
        data = []
        for i in f.readlines():
            curr_line = []
            if len(i) > 3:
                for j in i[0:-2].split(','):
                    curr_line.append(float(j[2:-1]))
            data.append(curr_line)
    return data

