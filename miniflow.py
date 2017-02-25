import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        #Nodes from which this node receives values
        self.inbound_nodes = inbound_nodes
        #Nodes to which thiss node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        #partials of this node w.r.t. input
        self.gradients = {}
        # for each inbound node here add this node as an outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
    def forward(self):
        #compute the output value based on 'inbound_nodes' and store the results in self.value
        pass
    def backward(self):
        pass


class Input(Node):
    def __init__(self):
        #input node has no inbound nodes
        Node.__init__(self)
        #it is the only node where the values can be passed as an argument to forward()
    def forward(self) :
        pass
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost*1

class Add(Node):
    #node that perform calculation: Addition
    #it takes two inbound nodes and addes the values of those nodes
    def __init__(self, *inputs):
        Node.__init__(self, inputs)
    def forward(self):
        self.value = 0
        for n in self.inbound_nodes:
            self.value += n.value
        #self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])
        
    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        Z = np.dot(X, W)
        self.value = Z + b
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    def _sigmoid(self, x):
        return 1./(1.+np.exp(-x))
    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])
    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y-a
        error = np.square(self.diff)
        self.value = np.mean(error)
    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

#def forward_pass(output_node, sorted_nodes):
#def forward_pass(graph):
def forward_and_backward(graph):
    #performs forward pass on list of sorted nodes and returns output_node's value
    for n in graph:
        n.forward()
    for n in graph[::-1]:
        n.backward()
    #return output_node.value

def sgd_update(trainable, learning_rate=1e-2):
    for t in trainable:
        partial = t.gradients[t]
        t.value -= learning_rate * partial