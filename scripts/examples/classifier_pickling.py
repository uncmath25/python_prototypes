import argparse
import itertools
### Necessary for virtual environment ###
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#########################################
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time


class ClassifierManager:
    """
    Loads, trains and predicts for a given classfier using pickling
    """

    def __init__(self, random_seed):
        """
        Expose internal objects
        """
        np.random.seed(int(random_seed))

        self._input_df = None
        self._FEATURE_NAMES = None

        self._pred_df = None
        self._pred_target = None

    def load_data(self, should_log):
        """
        Load input data
        """
        start_time = time.time()

        from sklearn.datasets import load_wine
        wine = load_wine()

        self._FEATURE_NAMES = tuple(wine.target_names)

        self._input_df = pd.DataFrame(wine.data, columns=[col_name for col_name in list(wine.feature_names)])
        self._input_df['TARGET'] = [self._FEATURE_NAMES[target_index] for target_index in list(wine.target)]

        self._input_df, pred_df = train_test_split(self._input_df, train_size=0.8, test_size=0.2)
        self._pred_df = pd.DataFrame(pred_df.iloc[:, :-1])
        self._pred_target = pd.Series(pred_df.iloc[:, -1])

        if should_log:
            print(wine.DESCR)
            print(self._input_df.shape)
            print(self._input_df.head())

        print('### LOADING THE DATA {0} TOOK SECONDS ###'.format(round(time.time() - start_time, 1)))

    def train_model(self, model_str, model_path, should_plot_cfm, cfm_plot_path, is_cfm_plot_normalized):
        """
        Train the model on the data based upon the specified model
        """
        train_df, test_df = train_test_split(self._input_df, train_size=0.8, test_size=0.2)
        train_X = train_df.iloc[:, :-1]
        train_y = [self._FEATURE_NAMES.index(target_class) for target_class in list(train_df.iloc[:, -1])] # factorized
        test_X = test_df.iloc[:, :-1]

        model = self._choose_model(model_str)
        model.fit(train_X, train_y)
        pickle.dump(model, open(model_path, 'wb'))

        preds = [int(pred) for pred in list(model.predict(test_X))] # factorized

        if should_plot_cfm:
            cfm = confusion_matrix(list(test_df.iloc[:, -1]), [self._FEATURE_NAMES[target_index] for target_index in preds]).tolist()
            print(np.array(cfm))
            self._plot_confusion_matrix(cfm, cfm_plot_path, is_cfm_plot_normalized)

    def _choose_model(self, model_str):
        """
        Initialize the model based upon the given model string
        """
        if model_str == 'lg':
            print('--- Using logisitic regression model ---')
            return LogisticRegression()
        elif model_str == 'rf':
            print('--- Using random forest model ---')
            return RandomForestClassifier()
        elif model_str == 'svm':
            print('--- Using support vector machine model ---')
            # return SVC(C=1, kernel='linear') # linear boundary
            return SVC(C=1, kernel='poly', degree=2) # non-linear boundary
            # return SVC(C=1, kernel='rbf')
            # return SVC(C=1, kernel='sigmoid') # binary classification

    def _plot_confusion_matrix(self, cfm, cfm_plot_path, is_cfm_plot_normalized):
        """
        Plots the given confusion matrix in a more appealing way
        """
        if is_cfm_plot_normalized:
            cfm_sum = sum([sum(row) for row in cfm])
            cfm = [[round(x / cfm_sum, 2) for x in row] for row in cfm]

        plt.rcParams["figure.figsize"] = (16, 9)

        plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        plt.xticks(list(range(len(self._FEATURE_NAMES))), self._FEATURE_NAMES)
        plt.yticks(list(range(len(self._FEATURE_NAMES))), self._FEATURE_NAMES)

        for i, j in itertools.product(range(len(cfm)), range(len(cfm[0]))):
            plt.text(j, i, '{:.2f}'.format(cfm[i][j]) if is_cfm_plot_normalized else '{:d}'.format(cfm[i][j]),
                        horizontalalignment='center', color='white' if (cfm[i][j]>max(max(cfm))/2) else 'black')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        title = 'Confusion Matrix'
        if is_cfm_plot_normalized:
            title = 'Normalized ' + title
        plt.title(title)

        plt.savefig(cfm_plot_path)
        plt.clf()

    def predict(self, model_path, results_path):
        """
        Predict using the model
        """
        model = pickle.load(open(model_path, 'rb'))
        pred_X = self._pred_df
        preds = [int(pred) for pred in list(model.predict(pred_X))] # factorized

        results_df = pd.DataFrame(list(zip([self._FEATURE_NAMES[target_index] for target_index in preds], self._pred_target)))
        results_df.index = list(self._pred_target.index)
        results_df.columns = ['Prediction', 'Actual']
        results_df.to_csv(results_path, header=True, index=False)


def run(model_str, output_dir):
    """
    Run the program using the cli inputs
    """
    RANDOM_SEED = 0

    SHOULD_LOG_IMPORT = True

    OUTPUT_DIR = str(output_dir)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    MODEL_STR = str(model_str)
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.pickle')
    SHOULD_PLOT_CFM = True
    CFM_PLOT_PATH = os.path.join(OUTPUT_DIR, 'cfm.png')
    IS_CFM_PLOT_NORMALIZED = False

    RESULTS_PATH = os.path.join(OUTPUT_DIR, 'results.csv')

    manager = ClassifierManager(RANDOM_SEED)
    manager.load_data(SHOULD_LOG_IMPORT)
    manager.train_model(MODEL_STR, MODEL_PATH, SHOULD_PLOT_CFM, CFM_PLOT_PATH, IS_CFM_PLOT_NORMALIZED)
    manager.predict(MODEL_PATH, RESULTS_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Pickling Example')
    parser.add_argument('model_str', help='Model Type String ("lg": Logistic Regression, "rf": Random Forest, "svm": Support Vector Machine)')
    parser.add_argument('output_dir', help='Directory where the results are outputted')
    args = parser.parse_args()

    try:
        run(args.model_str, args.output_dir)
    except Exception as e:
        print(e)
