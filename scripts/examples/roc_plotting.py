import argparse
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
### Necessary for virtual environment ###
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#########################################
import numpy as np
import os
import pandas as pd
import pickle
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
import time


class ROCManager():
    """
    Loads, trains and predicts for a given classfier and thens builds ROC curves
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

        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()

        self._FEATURE_NAMES = tuple(breast_cancer.target_names)

        self._input_df = pd.DataFrame(breast_cancer.data, columns=[col_name for col_name in list(breast_cancer.feature_names)])
        self._input_df['TARGET'] = [self._FEATURE_NAMES[target_index] for target_index in list(breast_cancer.target)]

        self._input_df, pred_df = train_test_split(self._input_df, train_size=0.8, test_size=0.2)
        self._pred_df = pd.DataFrame(pred_df.iloc[:, :-1])
        self._pred_target = pd.Series(pred_df.iloc[:, -1])

        if should_log:
            print(breast_cancer.DESCR)
            print(self._input_df.shape)
            print(self._input_df.head())

        print('### LOADING THE DATA {0} TOOK SECONDS ###'.format(round(time.time() - start_time, 1)))

    def plot_roc_distributions(self, model_str, resampling_number, roc_curve_steps, roc_plot_path):
        """
        Plot the ROC distributions for the given model
        """
        sampling_types = ['Normal', 'Oversampling', 'Undersampling']

        PLOT_MARGIN = 0.05
        plt.rcParams["figure.figsize"] = (16, 9)
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        sub_plot_index = 1

        for sampling_type in sampling_types:
            mean_fpr, mean_tpr, mean_threshold, mean_auc, std_auc = self._compute_mean_auc_data(sampling_type, model_str, resampling_number, roc_curve_steps)

            plt.subplot(int('22' + str(sub_plot_index)))

            sub_plot_index += 1

            plt.plot(mean_fpr, mean_tpr, color='g', label='AUC:{0}, STD:{1}'.format(round(mean_auc, 2), round(std_auc, 2)))
            plt.plot(mean_fpr, mean_threshold, linestyle='--', lw=2, color='b', label='Thresholds')
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance')

            plt.xlim([0-PLOT_MARGIN, 1+PLOT_MARGIN])
            plt.ylim([0-PLOT_MARGIN, 1+PLOT_MARGIN])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(sampling_type + ' ROC Distribution')
            plt.legend(loc="lower right")

        plt.savefig(roc_plot_path)
        plt.clf()

    def _compute_mean_auc_data(self, sampling_type, model_str, resampling_number, roc_curve_steps):
        """
        Compute the relevant data to plot a roc distribution
        """
        X = np.array(self._input_df.iloc[:, :-1])
        y = np.array([self._FEATURE_NAMES.index(target_class) for target_class in list(self._input_df.iloc[:, -1])]) # factorized

        model = self._choose_model(model_str)
        resampler = StratifiedKFold(resampling_number)

        mean_fpr = np.linspace(0, 1, roc_curve_steps)
        tprs = []
        thresholds = []
        aucs = []

        for train_indices, test_indices in resampler.split(X, y):
            if sampling_type == 'Oversampling':
                sampler = RandomOverSampler()
                X_train, y_train = sampler.fit_sample(X[train_indices], y[train_indices])
            elif sampling_type == 'Undersampling':
                sampler = RandomUnderSampler()
                X_train, y_train = sampler.fit_sample(X[train_indices], y[train_indices])
            else:
                sampler = RandomOverSampler()
                X_train, y_train = sampler.fit_sample(X[train_indices], y[train_indices])
            y_preds = model.fit(X_train, y_train).predict_proba(X[test_indices])[:, -1]
            fpr, tpr, threshold = roc_curve(y[test_indices], y_preds)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            thresholds.append(interp(mean_fpr, fpr, threshold))
            thresholds[-1][0] = 1.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_threshold = np.mean(thresholds, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        return(mean_fpr, mean_tpr, mean_threshold, mean_auc, std_auc)

    def _choose_model(self, model_str):
        """
        Initialize the model based upon the given model string
        """
        if model_str == 'lg':
            return(LogisticRegression())
        elif model_str == 'rf':
            return(RandomForestClassifier())
        elif model_str == 'svm':
            # return SVC(C=1, kernel='linear') # linear boundary
            return SVC(C=1, kernel='poly', degree=2) # non-linear boundary
            # return SVC(C=1, kernel='rbf')
            # return SVC(C=1, kernel='sigmoid') # binary classification


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
    RESAMPLING_NUMBER = 5
    ROC_CURVE_STEPS = 100
    ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, 'roc.png')

    manager = ROCManager(RANDOM_SEED)
    manager.load_data(SHOULD_LOG_IMPORT)
    manager.plot_roc_distributions(MODEL_STR, RESAMPLING_NUMBER, ROC_CURVE_STEPS, ROC_PLOT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC Plotting Example')
    parser.add_argument('model_str', help='Model Type String ("lg": Logistic Regression, "rf": Random Forest, "svm": Support Vector Machine)')
    parser.add_argument('output_dir', help='Directory where the results are outputted')
    args = parser.parse_args()

    try:
        run(args.model_str, args.output_dir)
    except Exception as e:
        print(e)
