import argparse
### Necessary for virtual environment ###
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#########################################
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.datasets
import sklearn.ensemble
import sklearn.impute
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree
import sys
import time


class MLPipeline():
    """
    Builds and runs a standard machine learning algorithm pipeline
    """

    def __init__(self, should_show_output):
        """
        Expose internal objects
        """
        self._SHOULD_SHOW_OUTPUT = bool(should_show_output)

        self._total_input_df = None
        self._train_data = None
        self._test_data = None
        self._train_X_arr = None
        self._train_y_arr = None
        self._test_X_arr = None
        self._test_y_arr = None

        self._input_pipeline = None

        self._model = None

    def load_input_data(self, target_hist_path):
        """
        Load sample input data
        """
        start_time = time.time()

        boston_data_bundle = sklearn.datasets.load_boston()
        if self._SHOULD_SHOW_OUTPUT:
            print(boston_data_bundle.DESCR)
            print('Dataset callables: {0}'.format(dir(boston_data_bundle)))

        self._total_input_df = pd.DataFrame(boston_data_bundle.data, columns=boston_data_bundle.feature_names)
        self._total_input_df['MHV'] = boston_data_bundle.target
        if self._SHOULD_SHOW_OUTPUT:
            print(self._total_input_df.info())
            print(self._total_input_df.describe())
            print(self._total_input_df.head())
        if self._SHOULD_SHOW_OUTPUT:
            print('Mean House Value Histogram:')
            self._total_input_df['MHV'].hist()
            plt.savefig(target_hist_path, format='png', dpi=100)
            plt.clf()

        print('### LOADING THE INPUT DATA TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def split_input_data(self, data_path):
        """
        Split the input data into training and test sets
        """
        start_time = time.time()

        # Keep track of only splitting indices if memory is an issue
        splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
        split = list(splitter.split(self._total_input_df, self._total_input_df['MHV'] >= 40))
        self._train_df = self._total_input_df.loc[list(split[0][0])]
        self._test_df = self._total_input_df.loc[list(split[0][1])]

        self._train_X_arr = self._train_df[[col for col in self._train_df.columns if col!='MHV']]
        self._train_y_arr = self._train_df['MHV']

        self._test_X_arr = self._test_df[[col for col in self._test_df.columns if col!='MHV']]
        self._test_y_arr = self._test_df['MHV']

        if self._SHOULD_SHOW_OUTPUT:
            print('Train DF Shape: {0}'.format(self._train_df.shape))
            print('Test DF Shape: {0}'.format(self._test_df.shape))
            print(self._train_df.head())

        # Temp way to generate sample data
        np.savetxt(data_path, self._train_X_arr[:100], delimiter=',')

        print('### SPLITTING THE INPUT DATA TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def investigate_input_data(self, corr_plot_path):
        """
        Investigate the input data
        """
        start_time = time.time()

        if self._SHOULD_SHOW_OUTPUT:
            corr_mat = self._train_df.corr().round(2)
            corr_mat['MHV'].sort_values(ascending=False)

        pd.plotting.scatter_matrix(self._train_df, figsize=(16, 16))
        plt.savefig(corr_plot_path, format='png', dpi=100)
        plt.clf()

        print('### INVESTIGATING THE INPUT DATA TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def clean_input_data(self):
        """
        Clean the input data with a resuable pipeline for the test data later
        """
        start_time = time.time()

        self._input_pipeline = sklearn.pipeline.Pipeline([
            ('imputer', sklearn.impute.SimpleImputer(strategy='median')),
            ('std_scaler', sklearn.preprocessing.StandardScaler())
        ])

        self._train_X_arr  = self._input_pipeline.fit_transform(self._train_X_arr)

        print('### CLEANING THE INPUT DATA TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def explore_models(self, models, scoring_metric='neg_mean_squared_error', cross_val_splits=10):
        """
        Explore different models on the input data using cross-validation
        """
        start_time = time.time()

        for model in models:
            sys.stderr = open(os.devnull, "w")
            scores = sklearn.model_selection.cross_val_score(model, self._train_X_arr, self._train_y_arr,
                                                             scoring=scoring_metric, cv=cross_val_splits)
            sys.stdout = sys.__stderr__
            scores = np.sqrt(-scores)

            print(str(model))
            print('Scores: {0}'.format([round(x, 2) for x in scores]))
            print('Mean: {0}'.format(round(scores.mean(), 2)))
            print('STD: {0}'.format(round(scores.std(), 2)))

        print('### EXPLORING MODELS TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def train_model(self, model, param_grid, model_pickle_path, scoring_metric='neg_mean_squared_error', cross_val_splits=10):
        """
        Train the optimized model using a grid search of the parameters
        """
        start_time = time.time()

        grid_search = sklearn.model_selection.GridSearchCV(model, param_grid, scoring=scoring_metric, cv=cross_val_splits)
        grid_search.fit(self._train_X_arr , self._train_y_arr)
        if self._SHOULD_SHOW_OUTPUT:
            print(grid_search.best_params_)

        self._model = grid_search.best_estimator_
        self._model.fit(self._train_X_arr , self._train_y_arr)

        if self._SHOULD_SHOW_OUTPUT:
            scores = sklearn.model_selection.cross_val_score(self._model, self._train_X_arr, self._train_y_arr,
                                                                scoring=scoring_metric, cv=cross_val_splits)
            scores = np.sqrt(-scores)

            print(str(self._model))
            print('Scores: {0}'.format([round(x, 2) for x in scores]))
            print('Mean: {0}'.format(round(scores.mean(), 2)))
            print('STD: {0}'.format(round(scores.std(), 2)))

        if self._SHOULD_SHOW_OUTPUT:
            print('Feature Importance')
            print(sorted(zip([col for col in self._train_df.columns if col!='MHV'],
                                [round(float(x), 2) for x in grid_search.best_estimator_.feature_importances_]), reverse=True))

        pickle.dump(self._model, open(model_pickle_path, 'wb'))

        print('### TRAINING THE MODELS TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def evaluate_model(self, model_eval_res_plot, plot_offset=1):
        """
        Evaluate the final trained model on the test set
        """
        start_time = time.time()

        self._test_X_arr = self._input_pipeline.transform(self._test_X_arr)
        pred_y_arr = self._model.predict(self._test_X_arr)

        print(pred_y_arr.shape)
        print('Actual: {0}'.format(self._test_y_arr[:3]))
        print('Predictions: {0}'.format(pred_y_arr[:3]))
        print('RMSE: {0}'.format(round(np.sqrt(sklearn.metrics.mean_squared_error(self._test_y_arr, pred_y_arr)), 2)))

        plt.rcParams["figure.figsize"] = (16, 9)

        PLOT_MIN = min(self._test_y_arr.min(), pred_y_arr.min()) - plot_offset
        PLOT_MAX = min(self._test_y_arr.max(), pred_y_arr.max()) + plot_offset

        plt.xlim(PLOT_MIN, PLOT_MAX)
        plt.ylim(PLOT_MIN, PLOT_MAX)

        plt.scatter(pred_y_arr, self._test_y_arr, color='blue', alpha=0.8, label='data')
        plt.plot([PLOT_MIN, PLOT_MAX], [PLOT_MIN, PLOT_MAX], color='green', alpha=0.8, label='fit')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Residues')
        plt.legend(loc='best')

        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig(model_eval_res_plot, format='png', dpi=100)
        # plt.savefig(model_eval_res_plot, format='png', dpi=100)
        plt.clf()

        print('### EVALUATING THE MODEL TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))

    def predict(self, model_pickle_path, data_path, pred_path):
        """
        Predict using the trained model at the given path using the specified data
        """
        start_time = time.time()

        model = pickle.load(open(model_pickle_path, 'rb'))

        data_X_arr = np.genfromtxt(data_path, delimiter=',')
        data_X_arr = self._input_pipeline.transform(data_X_arr)

        pred_y_arr = model.predict(self._test_X_arr)
        np.savetxt(pred_path, pred_y_arr, delimiter=',')

        print('### PREDICTING THE DATA TOOK {0} SECONDS ###'.format(round(time.time() - start_time, 1)))


def run(output_dir, should_show_output):
    """
    Run the program using the cli inputs
    """
    SHOULD_SHOW_OUTPUT = bool(should_show_output)

    OUTPUT_DIR = str(output_dir)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    TARGET_HIST_PATH = os.path.join(OUTPUT_DIR, 'target_hist.png')

    CORR_PLOT_PATH = os.path.join(OUTPUT_DIR, 'corr_plot.png')

    MODELS = [
        sklearn.linear_model.LinearRegression(),
        sklearn.tree.DecisionTreeRegressor(),
        sklearn.ensemble.RandomForestRegressor()
    ]

    MODEL = sklearn.ensemble.RandomForestRegressor()
    PARAM_GRID = [
        {'bootstrap': [True, False], 'max_features': [2, 5, 8, 12], 'n_estimators': [3, 10, 30]}
    ]
    MODEL_PICKLE_PATH = os.path.join(OUTPUT_DIR, 'model.pickle')

    MODEL_EVAL_RES_PLOT = os.path.join(OUTPUT_DIR, 'model_eval_res_plot.png')

    DATA_PATH = os.path.join(OUTPUT_DIR, 'sample_data.csv')
    PRED_PATH = os.path.join(OUTPUT_DIR, 'predictions.csv')

    pipeline = MLPipeline(SHOULD_SHOW_OUTPUT)
    pipeline.load_input_data(TARGET_HIST_PATH)
    pipeline.split_input_data(DATA_PATH)
    pipeline.investigate_input_data(CORR_PLOT_PATH)
    pipeline.clean_input_data()
    pipeline.explore_models(MODELS)
    pipeline.train_model(MODEL, PARAM_GRID, MODEL_PICKLE_PATH)
    pipeline.evaluate_model(MODEL_EVAL_RES_PLOT)
    pipeline.predict(MODEL_PICKLE_PATH, DATA_PATH, PRED_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline')
    parser.add_argument('output_dir', help='Directory where the results are outputted')
    parser.add_argument('-v', '--verbose', dest='should_show_output', action='store_true', help='Option to show output')
    args = parser.parse_args()

    try:
        run(args.output_dir, args.should_show_output)
    except Exception as e:
        print(e)
