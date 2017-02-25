import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        #Nodes from which this node receives values
        self.inbound_nodes = inbound_nodes
        #Nodes to which thiss node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # for each inbound node here add this node as an outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        
        
    def forward(self):
        #compute the output value based on 'inbound_nodes' and store the results in self.value
        pass

class Input(Node):
    def __init__(self):
        #input node has no inbound nodes
        Node.__init__(self)
        #it is the only node where the values can be passed as an argument to forward()
        def forward(self, value=None) :
            if value is not None:
                self.value = value

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

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    def _sigmoid(self, x):
        return 1./(1.+np.exp(-x))
    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])
    def forward(self):
        y = np.reshape(self.inbound_nodes[0].value, -1)
        a = np.reshape(self.inbound_nodes[1].value, -1)
        m = np.shape(self.inbound_nodes[0].value)
        error = np.square(y-a)
        self.value = np.mean(error)

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
def forward_pass(graph):
    #performs forward pass on list of sorted nodes and returns output_node's value
    for n in graph:
        n.forward()
    #return output_node.value