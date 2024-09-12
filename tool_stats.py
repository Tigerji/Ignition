def total_conn(conn_net):
    # number of connections in a connection map
    count = 0
    if conn_net:
        for i in conn_net:
            count = count + len(i)
    return count


def income_conn(conn_net):
    # return the total incoming connections
    income_net = [0 for _ in range(len(conn_net))]
    for i in range(len(conn_net)):
        if conn_net[i]:
            for j in conn_net[i]:
                income_net[j] = income_net[j] + 1
    return income_net


def is_strongly_connected(conn_net):
    # check whether a map is stongly connected
    num_nodes = len(conn_net)
    all_check = [False for _ in range(num_nodes)]
    for i in range(num_nodes):
        visited = [False for _ in range(num_nodes)]
        reached = [i]
        while reached:
            u = reached.pop()
            visited[u] = True
            for j in conn_net[u]:
                if not visited[j]:
                    reached.append(j)
        all_check[i] = all(visited)
    return all(all_check)


def shortest_length(conn_net):
    # get the average shortest path length for any two nodes in conn_net
    # use hash map larger amounts
    # can be used to check is_strongly_connected

    num_nodes = len(conn_net)
    aver_length = []
    max_length = []
    for i in range(num_nodes):
        lengths = [num_nodes + 1 for _ in range(num_nodes)]
        checking = [i]
        count = 0
        while checking:
            new_check = []
            for j in checking:
                if lengths[j] > count:
                    lengths[j] = count
                if conn_net[j]:
                    for s in conn_net[j]:
                        if lengths[s] > count + 1:
                            new_check.append(s)
            count = count + 1
            checking = new_check
        aver_length.append(sum(lengths) / num_nodes)
        max_length.append(max(lengths))
    return sum(aver_length) / num_nodes, max(max_length) < len(conn_net)


def clustering_coef_transitivity(conn_net):
    # clustering coefficient for conn_net
    # currently treating all maps as bi-directional
    total_triangle = len(conn_net) * (len(conn_net) - 1) * (len(conn_net) - 2) / 6
    count = 0
    for i in range(len(conn_net)):
        for j in range(i, len(conn_net)):
            for s in conn_net[i]:
                if s in conn_net[j]:
                    count = count + 1
    return count / total_triangle


def degree_distribution(conn_net):
    # presynaptic degree distribution
    connectivity = [0 for _ in conn_net]
    for i in conn_net:
        for j in i:
            connectivity[j] = connectivity[j] + 1
    distribution = [0 for _ in range(max(connectivity) + 1)]
    for i in connectivity:
        distribution[i] = distribution[i] + 1
    return distribution
