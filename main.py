import numpy as np
import sys
from ID3DecisionTree import ID3DecisionTree
# print(np.version.version)
# pandas.show_versions()


def main():
    if len(sys.argv) != 3:
        print('Invalid parameters')
        return 1

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    training_data = np.genfromtxt(train_file, encoding="utf8", names=True, dtype=int)
    testing_data = np.genfromtxt(test_file, encoding="utf8", names=True, dtype=int)

    # training_data2 = np.random.choice(training_data, size=100, replace=False)

    # training_data = np.concatenate([training_data, training_data2])
    # training_data = np.random.choice(training_data, size=100, replace=False)

    tree = ID3DecisionTree(training_data)
    tree.learn()
    tree.test_accuracy(training_data, 'training')
    print()
    tree.test_accuracy(testing_data, 'test')
    """
    training_data = np.genfromtxt("train.dat", encoding="utf8", names=True, dtype=int)
    testing_data = np.genfromtxt("test.dat", encoding="utf8", names=True, dtype=int)

    training_data2 = np.genfromtxt("train2.dat", encoding="utf8", names=True, dtype=int)
    testing_data2 = np.genfromtxt("test2.dat", encoding="utf8", names=True, dtype=int)

    training_data3 = np.genfromtxt("train3.dat", encoding="utf8", names=True, dtype=int)
    testing_data3 = np.genfromtxt("test3.dat", encoding="utf8", names=True, dtype=int)

    training_data4 = np.genfromtxt("train4.dat", encoding="utf8", names=True, dtype=int)
    testing_data4 = np.genfromtxt("test4.dat", encoding="utf8", names=True, dtype=int)

    tree = ID3DecisionTree(training_data)
    tree.learn()
    tree.test_accuracy(training_data, 'training')
    tree.test_accuracy(testing_data, 'test')

    tree2 = ID3DecisionTree(training_data2)
    tree2.learn()
    tree2.test_accuracy(training_data2, 'training')
    tree2.test_accuracy(testing_data2, 'test')

    tree3 = ID3DecisionTree(training_data3)
    tree3.learn()
    tree3.test_accuracy(training_data3, 'training')
    tree3.test_accuracy(testing_data3, 'test')

    tree4 = ID3DecisionTree(training_data4)
    tree4.learn()
    tree4.test_accuracy(training_data4, 'training')
    tree4.test_accuracy(testing_data4, 'test')

    tester = [111, 241, 102, 241, 163]
    best = max(range(len(tester)), key=tester.__getitem__)
    print(best)
    """
    return 0
if __name__ == '__main__':
    main()
