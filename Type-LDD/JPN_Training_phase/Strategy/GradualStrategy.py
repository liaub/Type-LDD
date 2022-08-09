'''
策略选择方式：
根据4类漂移，选择不同的策略进行学习
1.读取数据，根据类型训练贝叶斯网络
2.如果遇到drift point ，重新训练贝叶斯网络
3.遇到drift point ,根据类型策略选择重新学习贝叶斯网络
4:(1)abrupt 直接学习
(2) gradual 从第一个突变点学习
(3) incremental 从第一个漂移点与第一个突变点中间位置学习
'''
import skmultiflow
from pandas import DataFrame
from skmultiflow.bayes import NaiveBayes
import numpy as np
from skmultiflow.data import AGRAWALGenerator, LEDGenerator, SEAGenerator, HyperplaneGenerator


def generateGradualDriftStream(max_samples, first_func, second_func,
                              first_random_state, second_random_state,
                              all_random_state, drift_pos, window_len):

    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(stream=SEAGenerator(classification_function = first_func, random_state = first_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28),
                                        drift_stream=SEAGenerator(classification_function = second_func, random_state = second_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28), position=drift_pos,
                                        width=1000, random_state=all_random_state, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        iter_max_samples = max_samples / window_len
        iter_n_samples = 0
        correct_cnt = 0
        # Train the estimator with the samples provided by the data stream
        while iter_n_samples < iter_max_samples and stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred = naive_bayes.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            if n_samples<200:
                naive_bayes.partial_fit(X, y)
            iter_n_samples = iter_n_samples + 1
            n_samples += 1
        # Mark drift position
        resultList.append(correct_cnt / iter_n_samples)

    return resultList


def generateGradualDriftStream_No(max_samples, first_func, second_func,
                              first_random_state, second_random_state,
                              all_random_state, drift_pos, window_len):

    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(stream=SEAGenerator(classification_function = first_func, random_state = first_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28),
                                        drift_stream=SEAGenerator(classification_function = second_func, random_state = second_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28), position=drift_pos,
                                        width=1000, random_state=all_random_state, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    partial_samples = 0
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        if n_samples == drift_pos:  # 在这个点重新学习
            naive_bayes = NaiveBayes()
            partial_samples = 0
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                if partial_samples < 200:
                    naive_bayes.partial_fit(X, y)
                    partial_samples += 1
                # naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            # Mark drift position
            resultList.append(correct_cnt / iter_n_samples)
        else:
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                if partial_samples < 200:
                    naive_bayes.partial_fit(X, y)
                    partial_samples += 1
                # naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            # Mark drift position
            resultList.append(correct_cnt / iter_n_samples)

    return resultList


def generateGradualDriftStream_Yes(max_samples, first_func, second_func,
                              first_random_state, second_random_state,
                              all_random_state, drift_pos, window_len):

    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(stream=SEAGenerator(classification_function = first_func, random_state = first_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28),
                                        drift_stream=SEAGenerator(classification_function = second_func, random_state = second_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28), position=drift_pos,
                                        width=1000, random_state=all_random_state, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    partial_samples = 0
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        if n_samples == drift_pos+200:  # 在这个点重新学习
            naive_bayes = NaiveBayes()
            partial_samples = 0
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                if partial_samples < 200:
                    naive_bayes.partial_fit(X, y)
                    partial_samples += 1
                # naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            # Mark drift position
            resultList.append(correct_cnt / iter_n_samples)
        else:
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                if partial_samples < 200:
                    naive_bayes.partial_fit(X, y)
                    partial_samples += 1
                # naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            # Mark drift position
            resultList.append(correct_cnt / iter_n_samples)

    return resultList



if __name__ == "__main__":

    window_len = 51
    max_samples = 1275

    #Position range[5,95]
    position=[]
    for r in  range(50):
        if r%1==0:
            if r>=30 and r <40:
                position.append(r)
    print(len(position))

    errorbase = []
    # Gradual Drift （90*2*2*5*5）9000
    pair = [(2, 1)]
    for i in pair:
        first_func = i[0]
        second_func = i[1]
        for first_random_state in range(100, 101):  # (100, 112)
            for second_random_state in range(100, 110):  # (100, 112)
                for random_state in range(len(position)):
                    all_random_state = None
                    drift_pos = position[random_state] * 25
                    result = generateGradualDriftStream(max_samples, first_func, second_func, first_random_state,
                                                           second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
                    all_random_state = 112
                    drift_pos = position[random_state] * 25
                    result = generateGradualDriftStream(max_samples, first_func, second_func, first_random_state,
                                                           second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
    errorbase_ = np.mean(np.array(errorbase))
    print("baseline漂移策略的error 均值为：", errorbase_)
    errorbase = []

    errorbase = []
    #Gradual Drift （90*2*2*5*5）9000
    pair = [(2, 1)]
    for i in pair:
        first_func = i[0]
        second_func = i[1]
        for first_random_state in range(100, 101):#(100, 112)
            for second_random_state in range(100, 110):#(100, 112)
                for random_state in range(len(position)):
                    all_random_state = None
                    drift_pos = position[random_state]*25
                    result=generateGradualDriftStream_No(max_samples, first_func, second_func, first_random_state,
                                               second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
                    all_random_state = 112
                    drift_pos = position[random_state]*25
                    result=generateGradualDriftStream_No(max_samples, first_func, second_func, first_random_state,
                                               second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
    errorbase_ = np.mean(np.array(errorbase))
    print("未加入漂移策略的error 均值为：", errorbase_)
    errorbase = []

    pair = [(2, 1)]
    for i in pair:
        first_func = i[0]
        second_func = i[1]
        for first_random_state in range(100, 101):  # (100, 112)
            for second_random_state in range(100, 110):  # (100, 112)
                for random_state in range(len(position)):
                    all_random_state = None
                    drift_pos = position[random_state] * 25
                    result=generateGradualDriftStream_Yes(max_samples, first_func, second_func, first_random_state,
                                               second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
                    all_random_state = 112
                    drift_pos = position[random_state] * 25
                    result = generateGradualDriftStream_Yes(max_samples, first_func, second_func, first_random_state,
                                                        second_random_state, all_random_state, drift_pos, window_len)
                    errorbase = errorbase + result
    errorbase_ = np.mean(np.array(errorbase))
    print("加入漂移策略的error 均值为：", errorbase_)