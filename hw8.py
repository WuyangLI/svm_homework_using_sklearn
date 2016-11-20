from __future__ import division
from hw8utils import *

def q2():
    C = 0.01
    Q = 2
    [train_data, test_data] = load_raw_data()
    for i in range(0,10):
        print("label the data for digit "+str(i))
        [train_data_x, train_data_y, test_data_x, test_data_y] = label_data_one_vs_all(i, train_data, test_data)
        print("train the svm with poly kernel for digit "+str(i))
        my_svm_model = get_poly_kernel_svm(C, Q, train_data_x, train_data_y)
        print(str(i)+' : '+str(get_in_sample_error(my_svm_model, train_data_x, train_data_y)))

def q4():
    C = 0.01
    Q = 2
    [train_data, test_data] = load_raw_data()
    for i in [0,1]:
        print("label the data for digit "+str(i))
        [train_data_x, train_data_y, test_data_x, test_data_y] = label_data_one_vs_all(i, train_data, test_data)
        print("train the svm with poly kernel for digit "+str(i))
        my_svm_model = get_poly_kernel_svm(C, Q, train_data_x, train_data_y)
        print('number of support vectors for digit '+str(i)+ " : "+ str(sum(my_svm_model.n_support_)))

def q5():
    Q =2
    Carr = [0.001, 0.01, 0.1, 1]
    [train_data, test_data] = load_raw_data()
    [train_data_x, train_data_y, test_data_x, test_data_y] = label_data_one_vs_one(1, 5, train_data, test_data)
    for i in range(0,len(Carr)):
        print("train the svm with C equals "+str(Carr[i]))
        my_svm_model = get_poly_kernel_svm(Carr[i], Q, train_data_x, train_data_y)
        print('number of support vectors for '+str(Carr[i]) +' : '+str(sum(my_svm_model.n_support_)))
        print('Ein for '+str(Carr[i])+' : '+str(get_in_sample_error(my_svm_model, train_data_x, train_data_y)))
        print('Eout for '+str(Carr[i])+' : '+str(get_out_sample_error(my_svm_model, test_data_x, test_data_y)))

def q6():
    [train_data, test_data] = load_raw_data()
    [train_data_x, train_data_y, test_data_x, test_data_y] = label_data_one_vs_one(1, 5, train_data, test_data)
    Qarr = [2,5]
    Carr = [0.0001, 0.001, 0.01, 1]
    for i in range(0,len(Carr)):
        print('C equals '+str(Carr[i]))
        for j in range(0,len(Qarr)):
            print("  train the svm with C and Q : "+str(Carr[i])+' , '+str(Qarr[j]))
            my_svm_model = get_poly_kernel_svm(Carr[i], Qarr[j], train_data_x, train_data_y)
            print('    number of support vectors for '+str(Carr[i])+' , '+str(Qarr[j]) +' : '+str(sum(my_svm_model.n_support_)))
            print('    Ein for '+str(Carr[i])+' , '+str(Qarr[j])+' : '+str(get_in_sample_error(my_svm_model, train_data_x, train_data_y)))
            print('    Eout for '+str(Carr[i])+' , '+str(Qarr[j])+' : '+str(get_out_sample_error(my_svm_model, test_data_x, test_data_y)))
            
def q7():
    #cross validation
    [train_data, test_data] = load_raw_data()
    #filter out those labeled with 1 and 5 for this question
    train_data_filtered = filter(lambda x: x[0] == 1.0 or x[0] == 5.0, train_data)
    #the size of training data set is 1561, not a multiple of 10, let the whole dataset size be one off
    in_samples = train_data_filtered[1:]
    #parameters
    Q = 2
    Carr = [0.0001, 0.001, 0.01, 0.1, 1]
    #run for 100 times
    Ccount = {0.0001:0, 0.001:0, 0.01:0, 0.1:0, 1:0}
    Ecv_mean_all = 100*[[]]
    for i in range(0,100):
        #randomly shuffle the input data
	from random import shuffle
        shuffle(in_samples)
        #partition the data
        size = len(in_samples)/10
        Ecv = 10*[[]]
        for j in range(0,10):
            vs = int(j*size)
            ve = int((j+1)*size)
            validation = in_samples[vs:ve]
            from itertools import chain
            training_set = list(chain(in_samples[0:vs], in_samples[ve:]))
            training_x = map(lambda x : x[1:], training_set)
            training_y = map(lambda x : x[0], training_set)
            validation_x = map(lambda x : x[1:], validation)
            validation_y = map(lambda x : x[0], validation)
            my_svm_models = [get_poly_kernel_svm(C, Q, training_x, training_y) for C in Carr]
            Ecv[j] = [get_out_sample_error(my_svm_model, validation_x, validation_y) for my_svm_model in my_svm_models]
        #average the cv error
        Emean = [sum(e)/len(e) for e in zip(*Ecv)]
        #print(Emean)
        Ecv_mean_all[i] = Emean
        #choose the best C based on its corresponding Emean
        Cs = Carr[0]
        min = 100
        for k in range(0,len(Emean)):
            if Emean[k] < min:
                min = Emean[k]
                Cs = Carr[k]
        Ccount[Cs]+=1
    #print(Ecv_mean_all)
    Ecv_mean = [sum(e)/len(e) for e in zip(*Ecv_mean_all)]
    print("the counts of selected C")
    print(Ccount)
    print("average Ecv of all candidates")
    print(Ecv_mean)

def q9():
    [train_data, test_data] = load_raw_data()
    [train_data_x, train_data_y, test_data_x, test_data_y] = label_data_one_vs_one(1, 5, train_data, test_data)
    Carr = [0.01, 1, 100, 10**4, 10**6]
    for i in range(0,len(Carr)):
        print("  train the svm with C : "+str(Carr[i]))
        #K(x, y) = exp(-gamma ||x-y||^2)
        my_svm_model = svm.SVC(C=Carr[i],kernel='rbf', gamma = 1.0)
        my_svm_model.fit(train_data_x, train_data_y)
        print('    Ein for '+str(Carr[i])+' : '+str(get_in_sample_error(my_svm_model, train_data_x, train_data_y)))
        print('    Eout for '+str(Carr[i])+' : '+str(get_out_sample_error(my_svm_model, test_data_x, test_data_y)))


if __name__ == "__main__":
    q9()
