import math
import numpy as np


class ID3DecisionTree:
    def __init__(self, data):
        self.data = data
        self.root = None
        self.attributes = list(data.dtype.names[0:-1])
        self.className = data.dtype.names[-1]
        self.frequency = {}

    def information_gain(self, h, p, total_sample):
        return h - (self.entropy(p[0]) * len(p[0]) / total_sample + self.entropy(p[1]) *
                    len(p[1]) / total_sample + self.entropy(p[2]) * len(p[2]) / total_sample)

    def entropy(self, samples):
        if len(samples) == 0:
            return 0

        counts = [0, 0, 0]
        for row in samples:
            counts[row['class']] += 1
        h0 = h1 = h2 = 0
        p0 = counts[0] / len(samples)
        if p0 != 0:
            h0 = -p0 * math.log(p0, 2)
        p1 = counts[1] / len(samples)
        if p1 != 0:
            h1 = -p1 * math.log(p1, 2)
        p2 = counts[2] / len(samples)
        if p2 != 0:
            h2 = -p2 * math.log(p2, 2)

        return h0 + h1 + h2

    def partition(self, sample, attribute):
        partition = []
        for i in range(3):
            partition.append([])

        for val in sample:
            if val[attribute] == 0:
                partition[0].append(np.array(val))
            elif val[attribute] == 1:
                partition[1].append(np.array(val))
            else:
                partition[2].append(np.array(val))
        return partition

    # samples: rows of data
    def learn_tree(self, samples, unused_attributes, depth):
        # print(type(samples))
        node = Node()

        if len(samples) == 0:
            print(list(self.frequency.keys())[0])
            return None

        # if the sample is pure
        if len(set(samples[self.className])) == 1:
            node.data = samples[self.className][0]
            print(node.data)
            return node

        # if no attributes left to split
        if len(unused_attributes) == 0:
            count = [0, 0, 0]
            for i in range(len(samples)):
                count[int(samples[i][-1])] += 1

            node.data = max(range(len(count)), key=count.__getitem__)
            tie = []
            for i in range(len(count)):
                if count[i] == count[node.data]:
                    tie.append(i)
            if len(tie) != 1:
                maximum = list(self.frequency.keys())[0]
                if maximum not in tie:
                    maximum = list(self.frequency.keys())[1]
                node.data = maximum
            print(node.data)
            return node

        if depth != 0:
            print('')

        h = self.entropy(samples)
        attributes_ig = []
        for a in unused_attributes:
            p = self.partition(samples, a)
            total_sample = len(samples)
            ig = self.information_gain(h, p, total_sample)
            attributes_ig.append((a, ig, p))

        best = max(attributes_ig, key=lambda item: item[1])
        node.data = best[0]
        unused_attributes.remove(best[0])

        x = np.array(best[2][0])
        y = np.array(best[2][1])
        z = np.array(best[2][2])

        unused_attributes2 = [elem for elem in unused_attributes]
        for i in range(depth):
            print('| ', end='')
        print(node.data, '= 0 : ', end='')

        node.children[0] = self.learn_tree(x, unused_attributes2, depth + 1)
        for i in range(depth):
            print('| ', end='')
        print(node.data, '= 1 : ', end='')
        unused_attributes2 = [elem for elem in unused_attributes]
        node.children[1] = self.learn_tree(y, unused_attributes2, depth + 1)
        for i in range(depth):
            print('| ', end='')
        print(node.data, '= 2 : ', end='')

        unused_attributes2 = [elem for elem in unused_attributes]
        node.children[2] = self.learn_tree(z, unused_attributes2, depth + 1)

        return node

    def predict(self, node, row):
        if node is None:
            return list(self.frequency.keys())[0]
        if node.data == 0 or node.data == 1 or node.data == 2:
            return node.data
        else:
            attribute = node.data
            val = row[attribute]
            return self.predict(node.children[val], row)  # goto next feature

    def test_accuracy(self, data, description):
        hit = 0
        for row in data:
            prediction = self.predict(self.root, row)
            if prediction == row[self.className]:
                hit += 1
        print('Accuracy on {0} set ({1} instances): {2}%'.format(description, len(data),
                                                                 str(round(100 * hit / len(data), 1))))

    def learn(self):
        # get the most frequent class value in the set
        count = {0: 0, 1: 0, 2: 0}
        for i in self.data:
            count[i[-1]] += 1

        sorted_values = sorted(count.values())  # Sort the values
        sorted_values.reverse()
        sorted_dict = {}

        for i in sorted_values:
            for k in count.keys():
                if count[k] == i:
                    sorted_dict[k] = count[k]
                    break

        self.frequency = sorted_dict

        # call learning algorithm
        self.root = self.learn_tree(self.data, self.attributes, 0)
        print()


class Node:
    def __init__(self, data=None):
        self.data = data
        self.children = [None, None, None]
