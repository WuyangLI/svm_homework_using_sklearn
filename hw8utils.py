from sklearn import svm

def load_raw_data():
    train_raw = open('features.train', 'r')
    train_data = [map(float, line.strip().split('  ')) for line in train_raw]

    test_raw = open('features.test', 'r')
    test_data = [map(float, line.strip().split('  ')) for line in test_raw]

    return [train_data, test_data]

def get_poly_kernel_svm(C_in, Q_in, train_data_x, train_data_y):
    #for poly kernel, (gamma * u'* v + coef0)^degree
    my_svm_model = svm.SVC(C=C_in,kernel='poly', degree=Q_in, gamma = 1.0, coef0=1.0)
    return my_svm_model.fit(train_data_x, train_data_y)

def label_data_one_vs_all(digit, train_data, test_data):
    train_data_x = map(lambda x: x[1:], train_data)
    train_data_y = map(lambda x: 1 if x[0] == digit else -1, train_data)
    test_data_x = map(lambda x: x[1:], test_data)
    test_data_y = map(lambda x: 1 if x[0] == digit else -1, test_data)
    return [train_data_x, train_data_y, test_data_x, test_data_y]

def label_data_one_vs_one(digit1, digit2, train_data, test_data):
    #filter out data samples for the input two digits
    train_data_filtered = filter(lambda x: x[0] == digit1 or x[0] == digit2, train_data)
    test_data_filtered = filter(lambda x: x[0] == digit1 or x[0] == digit2, test_data)
    #label the data
    train_data_x = map(lambda x: x[1:], train_data_filtered)
    train_data_y = map(lambda x: 1 if x[0] == digit1 else -1, train_data_filtered)
    test_data_x = map(lambda x: x[1:], test_data_filtered)
    test_data_y = map(lambda x: 1 if x[0] == digit1 else -1, test_data_filtered)
    return [train_data_x, train_data_y, test_data_x, test_data_y]    

def get_in_sample_error(my_svm_model, train_data_x, train_data_y):
    return 1-my_svm_model.score(train_data_x, train_data_y)

def get_out_sample_error(my_svm_model, test_data_x, test_data_y):
   return 1-my_svm_model.score(test_data_x, test_data_y)

    
