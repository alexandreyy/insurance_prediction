'''
Created on 15/11/2015

@author: Alexandre Yukio Yamashita
@author: Celso Kakihara
'''

from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.label import LabelEncoder
from unbalanced_dataset.over_sampling import OverSampler, SMOTE
from unbalanced_dataset.under_sampling import TomekLinks

import numpy as np
import pandas as pd


class Data:
    '''
    Read data.
    '''

    def __init__(self, train_path = "", test_path = ""):
        # Load data if path is set.
        if train_path != "" and test_path != "":
            self.load(train_path, test_path)
        else:
            self.train_x = None
            self.train_y = None
            self.validation_x = None
            self.validation_y = None
            self.test_x = None


    def load(self, train_path, test_path):
        '''
        Load data.
        '''
        # Load data.
        print "Loading data."
        self.train_x = pd.read_csv(train_path)
        self.train_x = self.train_x.as_matrix()
        self.train_y = self.train_x[:, 2].astype(np.bool)
        self.train_x = np.hstack((self.train_x[:, 1:2], self.train_x[:, 3:]))

        self.test_x = pd.read_csv(test_path)
        self.test_x = self.test_x.as_matrix()
        self.test_x = self.test_x[:, 1:]


    def construct_features(self):
        '''
        Construct features.
        '''
        # Parse date features.
        print "Parsing date features"
        parsed_train_X = self.parse_date_feature(self.train_x[:, 0])
        parsed_test_X = self.parse_date_feature(self.test_x[:, 0])

        # Parse other features.
        print "Parsing all features"
        total_train = len(self.train_x)
        total_test = len(self.test_x)

        for index_feature in range(1, len(self.train_x[0])):
            print "Processing feature ", index_feature

            # Check if we have a categorical feature.
            labels = np.unique(self.train_x[:, index_feature])

            # If we have string or binary labels, we have a categorical feature.
            if type(self.train_x[0, index_feature]) == np.str or len(labels) == 2:
                # We have a categorical feature.

                # Encode it in the one hot format.
                original_data = np.hstack((self.train_x[:, index_feature],
                                           self.test_x[:, index_feature]))

                label_encoder = LabelEncoder()
                data_label_encoded = label_encoder.fit_transform(original_data)
                encoder = OneHotEncoder()
                data_encoded = encoder.fit_transform(data_label_encoded.reshape((len(data_label_encoded), 1)))
                data_encoded = np.asarray(data_encoded.todense()).astype(np.bool)

                # Add encoded feature to data.
                parsed_train_X = np.hstack((parsed_train_X, data_encoded[0:total_train, :]))
                parsed_test_X = np.hstack((parsed_test_X, data_encoded[total_train:, :]))
                del data_encoded
            else:
                # We have a numeric feature.

                # Just add it to the data.
                parsed_train_X = np.hstack((parsed_train_X,
                                            self.train_x[:, index_feature].reshape((total_train, 1))))
                parsed_test_X = np.hstack((parsed_test_X,
                                           self.test_x[:, index_feature].reshape((total_test, 1))))

        self.train_x = parsed_train_X
        self.test_x = parsed_test_X


    def order_binary_features(self):
        '''
        Move binary features to first columns.
        '''
        binary_features = []

        # Find binary features.
        for index_feature in range(0, len(self.train_x[0])):
            binary_features.append(type(self.train_x[0, index_feature]) == np.bool)

        print np.sum(binary_features)
        binary_features = np.array(binary_features)
        binary_query = np.where(binary_features)
        non_binary_query = np.where(np.not_equal(binary_features, True))

        total_train = len(self.train_x)
        total_test = len(self.test_x)
        total_binary_features = np.sum(binary_features)
        total_non_binary_features = len(binary_features) - np.sum(binary_features)

        # Order features.
        self.train_x = np.hstack((self.train_x[:, binary_query].reshape((total_train, total_binary_features)),
                                  self.train_x[:, non_binary_query].reshape((total_train, total_non_binary_features))))
        self.test_x = np.hstack((self.test_x[:, binary_query].reshape((total_test, total_binary_features)),
                                 self.test_x[:, non_binary_query].reshape((total_test, total_non_binary_features))))


    def replace_nan_features(self):
        '''
        Replace nan features.
        '''
        print "Replacing nan features."

        # For all features, replace nan values.
        for index_feature in range(self.train_x.shape[1]):
            train_null_query = pd.isnull(self.train_x[:, index_feature])
            train_non_null_query = np.not_equal(train_null_query, True)
            train_null_query = np.where(train_null_query)[0]
            train_non_null_query = np.where(train_non_null_query)[0]

            test_null_query = pd.isnull(self.test_x[:, index_feature])
            test_non_null_query = np.not_equal(test_null_query, True)
            test_null_query = np.where(test_null_query)[0]
            test_non_null_query = np.where(test_non_null_query)[0]

            # Check if we found nan values.
            if len(train_null_query) > 0 or len(test_null_query) > 0:
                # We found nan values.

                # Get labels from feature.
                labels = np.unique(self.train_x[train_non_null_query, index_feature])

                # If we have two labels, we have a binary feature.
                if len(labels) == 2:
                    # We have a binary feature.

                    # Check if data is a yes/no or 0/1 feature.
                    if type(labels[0]) != np.str:
                        # We have a 0/1 feature.

                        # Replace value with the more common answer.
                        total_1 = len(np.where(np.equal(self.train_x[train_non_null_query, index_feature], 1.0))[0])
                        total_0 = abs(len(train_non_null_query) - total_1)

                        if total_1 > total_0:
                            replace_by = 1
                        else:
                            replace_by = 0
                    else:
                        # We have a yes/no feature.

                        # Replace value with the more common answer.
                        total_yes = len(np.where(np.equal(self.train_x[train_non_null_query, index_feature], "Y"))[0])
                        total_no = abs(len(train_non_null_query) - total_yes)

                        if total_yes > total_no:
                            replace_by = "Y"
                        else:
                            replace_by = "N"
                else:
                    # Replace feature by mean.
                    replace_by = np.mean(self.train_x[train_non_null_query, index_feature])
                    replace_by = int(replace_by + 0.5)

                if len(train_null_query) > 0:
                    self.train_x[train_null_query, index_feature] = replace_by

                if len(test_null_query) > 0:
                    self.test_x[test_null_query, index_feature] = replace_by


    def balance_data_oversampling(self, ratio = 2, balance_type = "OverSampler"):
        '''
        Balance data.
        '''
        verbose = True

        if balance_type == "OverSampler":
            sm = OverSampler(verbose = verbose, ratio = ratio)
        elif balance_type == 'SMOTE_borderline1':
            sm = SMOTE(kind = 'borderline1', verbose = verbose, ratio = ratio)
        elif balance_type == 'SMOTE_regular':
            sm = SMOTE(kind = 'regular', verbose = verbose, ratio = ratio)
        elif balance_type == 'SMOTE_borderline2':
            sm = SMOTE(kind = 'borderline2', verbose = verbose, ratio = ratio)
        else:
            sm = TomekLinks(verbose = verbose)

        self.train_x, self.train_y = sm.fit_transform(self.train_x, self.train_y)


    def parse_date_feature(self, data):
        '''
        Parse date feature.
        '''
        years = []
        months = []
        days = []

        for index_feature in range(len(data)):
            data_parsed = data[index_feature].split("-")
            years.append(int(data_parsed[0]))
            months.append(int(data_parsed[1]))
            days.append(int(data_parsed[2]))

        years = np.array(years)
        months = np.array(months)
        days = np.array(months)
        date_feature = np.vstack((years, months, days))
        date_feature = np.transpose(date_feature)

        return date_feature


    def save(self, path_data):
        '''
        Save pre-processed data;
        '''
        print "Saving data."

        f = file(path_data, "wb")
        np.save(f, self.train_x)
        np.save(f, self.train_y)
        np.save(f, self.test_x)
        f.close()


    def load_parsed_data(self, path_data, one_hot = False):
        '''
        Load parsed data.
        '''
        print "Loading parsed data."

        f = file(path_data, "rb")
        self.train_x = np.load(f)
        self.train_y = np.load(f)
        self.test_x = np.load(f)
        f.close()

        if one_hot:
            n = 2
            self.train_y = self.train_y.flatten()
            o_h = np.zeros((len(self.train_y), n))
            o_h[np.arange(len(self.train_y)), self.train_y.astype(int)] = 1
            self.train_y = o_h.astype(bool)


    def save_sliptted_data(self, path_data):
        '''
        Save splitted data;
        '''
        print "Saving data."

        f = file(path_data, "wb")
        np.save(f, self.train_x)
        np.save(f, self.train_y)
        np.save(f, self.validation_x)
        np.save(f, self.validation_y)
        np.save(f, self.test_x)
        f.close()


    def load_sliptted_data(self, path_data, one_hot = False):
        '''
        Load splitted data;
        '''
        print "Loading data."

        f = file(path_data, "rb")
        self.train_x = np.load(f)
        self.train_y = np.load(f)
        self.validation_x = np.load(f)
        self.validation_y = np.load(f)
        self.test_x = np.load(f)
        f.close()

        if one_hot:
            n = 2
            self.train_y = self.train_y.flatten()
            o_h = np.zeros((len(self.train_y), n))
            o_h[np.arange(len(self.train_y)), self.train_y.astype(int)] = 1
            self.train_y = o_h.astype(bool)
            self.validation_y = self.validation_y.flatten()
            o_h = np.zeros((len(self.validation_y), n))
            o_h[np.arange(len(self.validation_y)), self.validation_y.astype(int)] = 1
            self.validation_y = o_h.astype(bool)

        indexes = range(len(self.train_x))
        np.random.shuffle(indexes)
        self.train_x = self.train_x[indexes, :]
        self.train_y = self.train_y[indexes]

        indexes = range(len(self.validation_x))
        np.random.shuffle(indexes)
        self.validation_x = self.validation_x[indexes, :]
        self.validation_y = self.validation_y[indexes]


    def z_norm_by_feature(self):
        '''
        Normalize data using z-norm.
        '''

        print "Normalizing features."
        binary_index = 384
        X = np.vstack((self.train_x[:, binary_index:].astype(np.float),
                       self.validation_x[:, binary_index:].astype(np.float)))

        # Calculate mean and std, if needed.
        mean_value = np.mean(X, axis = 0)
        std_value = np.std(X, axis = 0)
        std_value[np.where(np.equal(std_value, 0))] = 1
        del X

        # Normalize data.
        self.train_x[:, binary_index:] = np.divide(np.subtract(self.train_x[:, binary_index:], mean_value),
                                                   np.array(std_value))
        self.validation_x[:, binary_index:] = np.divide(np.subtract(self.validation_x[:, binary_index:], mean_value),
                                                        np.array(std_value))
        self.test_x[:, binary_index:] = np.divide(np.subtract(self.test_x[:, binary_index:], mean_value),
                                                  np.array(std_value))

    def z_norm_train_test_by_feature(self):
        '''
        Normalize data using z-norm.
        '''

        print "Normalizing features."
        binary_index = 384
        X = self.train_x[:, binary_index:].astype(np.float)

        # Calculate mean and std, if needed.
        mean_value = np.mean(X, axis = 0)
        std_value = np.std(X, axis = 0)
        std_value[np.where(np.equal(std_value, 0))] = 1
        del X

        # Normalize data.
        self.train_x[:, binary_index:] = np.divide(np.subtract(self.train_x[:, binary_index:], mean_value),
                                                   np.array(std_value))
        self.test_x[:, binary_index:] = np.divide(np.subtract(self.test_x[:, binary_index:], mean_value),
                                                  np.array(std_value))

if __name__ == '__main__':
    '''
    Pre-process data.
    '''

    train_path = "../resources/train.csv"
    test_path = "../resources/test.csv"
    save_path = "../resources/parsed_data.bin"

#     homesite = Data(train_path, test_path)
#     print homesite.train_x.shape
#     homesite.replace_nan_features()
#     homesite.construct_features()
#     homesite.order_binary_features()
#     homesite.save(save_path)
#     print homesite.train_x.shape, homesite.train_y.shape, homesite.test_x.shape

    load_path = "../resources/parsed_data.bin"
    homesite = Data()
    homesite.load_parsed_data(load_path)
    print homesite.train_x.shape, homesite.train_y.shape, homesite.test_x.shape
