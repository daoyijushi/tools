import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from obtain_standard_input import standard_input
from sklearn.externals import joblib
from evaluate import calculate_indicator
import matplotlib.pyplot as plt

# clf = SVC(decision_function_shape='ovo')



def classifier(train_X, train_Y):
    """
    创建分类器
    :param X: input data
    :param Y: classes
    :return: predict
    """

    # clf = SVC() # 结果是46%
    # clf = SVC(decision_function_shape='ovr') # 结果是46%
    clf = LinearSVC()
    clf.fit(train_X, train_Y)

    return clf


def data_process(path):
    """
    数据处理
    :param path: 数据路径
    :return: train_X_L, train_Y_L, train_X_R, train_Y_R, test_X, test_Y
    """

    dataset = standard_input(path)
    datasets = shuffle(dataset, random_state=0)
    data = []
    target = []
    train_X_L = []
    train_Y_L = []
    train_X_R = []
    train_Y_R = []


    for item in dataset:
        #print(item)
        data.append(item[0:4])
        target.append(item[-1])
    # print(data)
    # print(target)

    train_data = data[0:2000]
    train_target = target[0:2000]

    test_X = data[2000:]
    test_Y = target[2000:]

    for i in range(len(train_data)):
        if train_data[i][1] >= 256:
            train_X_L.append(train_data[i])
            train_Y_L.append(target[i])
        elif train_data[i][1] < 256:
            train_X_R.append(train_data[i])
            train_Y_R.append(target[i])
        else:
            continue
    return train_X_L, train_Y_L, train_X_R, train_Y_R, test_X, test_Y

def predict(test_X, clf_L, clf_R):
    """

    :param tese_X:
    :param clf_L:
    :param clf_R:
    :param pre_y:
    :return:
    """
    pre_y = []
    for i in range(len(test_X)):
        if test_X[i][1] >= 256:
            pre_y.append(clf_L.predict(np.array(test_X[i], dtype=float).reshape(1, -1)))
        elif test_X[i][1] < 256:
            pre_y.append(clf_R.predict(np.array(test_X[i], dtype=float).reshape(1, -1)))

    return pre_y

def acc(test_Y, pre_y):
    """

    :param test_Y: groundtruth
    :param pre_y:  predict
    :return:
    """
    count = 0
    for i in range(len(pre_y)):
        # print(pre_y[i], test_Y[i])
        if pre_y[i] == test_Y[i]:
            count += 1

    print(count / len(pre_y))
    return (count / len(pre_y))


def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    path = "fulldata.txt"
    train_X_L, train_Y_L, train_X_R, train_Y_R, test_X, test_Y = data_process(path)
    clf_L = classifier(train_X_L, train_Y_L)
    clf_R = classifier(train_X_R, train_Y_R)
    pre_y = predict(test_X, clf_L, clf_R)
    pre_acc = acc(test_Y, pre_y)
    calculate_indicator(test_Y, pre_y)

    joblib.dump(clf_L, "model_L_" + str(pre_acc) + ".m")
    joblib.dump(clf_R, "model_R_" + str(pre_acc) + ".m")

    labels = list(set(test_Y))
    plot_confusion_matrix(test_Y, pre_y, labels)

