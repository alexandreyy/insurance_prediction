'''
Created on 25/11/2015

@author: Alexandre Yukio Yamashita
'''

from theano import tensor as T
import theano

from data.data import Data
from statistics.confusion_matrix import confusion_matrix
from statistics.performance import compute_auc, compute_performance_metrics
import numpy as np


class NeuralNetwork:
    '''
    Theano neural network.
    '''

    def __init__(self, path = "", input_units = 1, hidden_units = 1, output_units = 1,
                 lr = 0.05, lamb = 0.000005):
        X = T.fmatrix()
        Y = T.fmatrix()

        # Initialze weights.
        if path == "":
            self.w_h = self._initialize_weights((input_units, hidden_units))
            self.w_o = self._initialize_weights((hidden_units, output_units))
        else:
            self.load(path)

        # Define ANN model.
        py_x = self._model(X, self.w_h, self.w_o)
        hidden_output = self._model_2(X, self.w_h)
        y_x = T.argmax(py_x, axis = 1)

        # Define cost and train method.
        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
#         cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) + \
#                      lamb * (T.sum(T.sqr(self.w_h) + T.sum(T.sqr(self.w_o))))
        params = [self.w_h, self.w_o]
        updates = self._sgd(cost, params, lr)

        # Define train predict function.
        self._t_train = theano.function(inputs = [X, Y], outputs = cost, updates = updates, allow_input_downcast = True)
        self._t_predict_proba = theano.function(inputs = [X], outputs = py_x, allow_input_downcast = True)
        self._t_predict = theano.function(inputs = [X], outputs = y_x, allow_input_downcast = True)
        self._t_hidden_output = theano.function(inputs = [X], outputs = hidden_output, allow_input_downcast = True)


    def fit(self, data, batch_size = 128,
            max_iterations = 100, save_interval = 10, path = "ann_weights.bin", return_cost = False):
        '''
        Train neural network.
        '''

        cost_history = []
        self.best_w_h = self.w_h.get_value()
        self.best_w_o = self.w_o.get_value()
        best_auc = 0

        for iteration in range(max_iterations):
            i = 0

            for start, end in zip(range(0, len(data.train_x), batch_size), range(batch_size, len(data.train_x), batch_size)):

                cost = self._t_train(data.train_x[start:end], data.train_y[start:end])
                i = i + 1

                if i % save_interval == 0:
                    self.save(path)

                    if data.validation_y is not None:
                        predicted_labels = self.predict_proba(data.validation_x)[:, 1]

                        auc = compute_auc(np.argmax(data.validation_y, axis = 1), predicted_labels)

                        if auc > best_auc:
                            best_auc = auc
                            self.best_w_h = self.w_h.get_value()
                            self.best_w_o = self.w_o.get_value()
                        else:
                            if abs(best_auc - auc) < 0.000005:
                                self.w_h.set_value(self.best_w_h)
                                self.w_o.set_value(self.best_w_o)
                                return

                    self.save(path)

            print cost

            if return_cost:
                cost_history.append(cost)
                print cost

        return np.array(cost_history)


    def predict(self, x):
        '''
        Predict label.
        '''

        return self._t_predict(x)


    def predict_proba(self, x):
        '''
        Predict probability.
        '''

        return self._t_predict_proba(x)


    def get_hidden_output(self, x):
        '''
        Predict label.
        '''

        return self._t_hidden_output(x)


    def _floatX(self, X):
        '''
        Convert numpy array to float.
        '''

        return np.asarray(X, dtype = theano.config.floatX)  # @UndefinedVariable


    def _initialize_weights(self, shape):
        '''
        Initialize neural network weights.
        '''

        return theano.shared(self._floatX(np.random.randn(*shape) * 0.01))


    def _sgd(self, cost, params, lr = 0.05):
        '''
        Train ANN using SGD.
        '''

        grads = T.grad(cost = cost, wrt = params)
        updates = []

        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates


    def _model(self, X, w_h, w_o):
        '''
        ANN model.
        '''

        h = T.nnet.sigmoid(T.dot(X, w_h))
        pyx = T.nnet.sigmoid(T.dot(h, w_o))

        return pyx


    def _model_2(self, X, w_h):
        '''
        ANN model 2.
        '''

        h = T.nnet.sigmoid(T.dot(X, w_h))

        return h


    def save(self, path):
        '''
        Save ANN weights.
        '''

        f = file(path, "wb")
        np.save(f, self.w_h.get_value())
        np.save(f, self.w_o.get_value())
        f.close()


    def load(self, path):
        '''
        Load ANN weights;
        '''

        f = file(path, "rb")
        self.w_h = np.load(f)
        self.w_o = np.load(f)
        f.close()

        self.w_h = theano.shared(self._floatX(self.w_h))
        self.w_o = theano.shared(self._floatX(self.w_o))


if __name__ == '__main__':
    '''
    Train neural network.
    '''

    # oversampled_path = "../../homesite_data/resources/oversampled_normalized_data_ratio_2.5.bin"
    oversampled_path = "../../homesite_data/resources/oversampled_normalized_data_ratio_2.bin"
    homesite_data = Data()
    homesite_data.load_sliptted_data(oversampled_path, one_hot = True)

    # Train neural network.
    clf = NeuralNetwork(input_units = 644, hidden_units = 50, output_units = 2, \
                        lr = 0.00005, lamb = 0.)
#     clf.fit(homesite_data, batch_size = 128,
#             max_iterations = 100, save_interval = 10,
#             path = "../homesite_data/ann_weights.bin")

    # Test neural network.
#     clf = NeuralNetwork(path = "../../homesite_data/ann_weights.bin", lr = 0.05, lamb = 0.000005)

    # Test classifier.
    print 'Testing classifier.'
    predicted_labels = clf.predict_proba(homesite_data.validation_x)[:, 1]

    # Show final results.
    results = confusion_matrix(np.argmax(homesite_data.validation_y, axis = 1), np.round(predicted_labels))
    accuracy, precision, recall = compute_performance_metrics(results)
    auc = compute_auc(np.argmax(homesite_data.validation_y, axis = 1), predicted_labels)
