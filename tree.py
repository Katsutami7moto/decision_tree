import math


def majority_value(data, target_attr):
    val_freq = {}
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1
        else:
            val_freq[record[target_attr]] = 1
    return max(val_freq, key=val_freq.get)


def choose_attribute(data, attributes, target_attr, fitness_func):
    gains = {}
    for attr, name in enumerate(attributes):
        if attr == target_attr:
            continue
        else:
            gains[name] = gain(data, attr, target_attr)
    return fitness_func(gains, key=gains.get)


def get_values(data, best):
    return {record[index_names[best]] for record in data}


def get_examples(data, best, val):
    return [record for record in data if record[index_names[best]] == val]


def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if (len(attributes) - 1) <= 0:
        return Node(default)
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return Node(vals[0])
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = Node(best)

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = Node(val, tree)
            subtree.children.append(create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func))

    return tree


def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if record[attr] in val_freq:
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return entropy(data, target_attr) - subset_entropy


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)


def print_tree(current_node, indent="", last='updown'):
    def nb_children(node):
        return sum(nb_children(child) for child in node.children) + 1

    size_branch = {child: nb_children(child) for child in current_node.children}

    """ Creation of balanced lists for "up" branch and "down" branch. """
    up = sorted(current_node.children, key=lambda node: nb_children(node))
    down = []
    while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
        down.append(up.pop())

    """ Printing of "up" branch. """
    for child in up:
        next_last = 'up' if up.index(child) is 0 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', " " * len(current_node.name))
        print_tree(child, indent=next_indent, last=next_last)

    """ Printing of current node. """
    if last == 'up':
        start_shape = '┌'
    elif last == 'down':
        start_shape = '└'
    elif last == 'updown':
        start_shape = ' '
    else:
        start_shape = '├'

    if up:
        end_shape = '┤'
    elif down:
        end_shape = '┐'
    else:
        end_shape = ''

    print('{0}{1}{2}{3}'.format(indent, start_shape, current_node.name, end_shape))

    """ Printing of "down" branch. """
    for child in down:
        next_last = 'down' if down.index(child) is len(down) - 1 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', " " * len(current_node.name))
        print_tree(child, indent=next_indent, last=next_last)


def read_data_from_file():
    all_input = []
    input_file = open('table.txt', 'r')
    for line in input_file:
        all_input.append(line.split())
    input_file.close()
    attributes = all_input[0]
    data = all_input[1:]
    return attributes, data


if __name__ == '__main__':
    data_to_go = read_data_from_file()
    index_names = dict(zip(data_to_go[0], range(len(data_to_go[0]))))
    print_tree(create_decision_tree(data_to_go[1], data_to_go[0], 0, max))
    input()
