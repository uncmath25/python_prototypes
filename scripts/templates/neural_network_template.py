import argparse
import csv
from keras.layers import Dense, Dropout
from keras.models import Sequential
### Necessary for virtual environment ###
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#########################################
import numpy as np
import os
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
import sys
import time


class NeuralNetworkManager():
    """
    Template for building, training and predicting with a neural network
    """

    def __init__(self, temp_predictions_path, sql_username, sql_password, database_ip_address, database_port_number):
        """
        Initializes sql credentials and attribute information
        """
        np.random.seed(1)

        self._TEMP_PREDICTIONS_PATH = str(temp_predictions_path)

        self._SQL_USERNAME = str(sql_username)
        self._SQL_PASSWORD = str(sql_password)
        self._DATABASE_IP_ADDRESS = str(database_ip_address)
        self._DATABASE_PORT_NUMBER = int(database_port_number)

        self._input_df = None
        self._model = None
        self._pred_df = None

    def _query_database(self, query):
        """
        Queries the aurora database and returns any fetched data
        """
        conn = pymysql.connect(user=self._SQL_USERNAME, passwd=self._SQL_PASSWORD, host=self._DATABASE_IP_ADDRESS, port=self._DATABASE_PORT_NUMBER, charset='utf8', local_infile=True)

        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        if cur.description is not None:
            column_names = [str(row[0]) for row in cur.description]
        else:
            column_names = None

        cur.close()
        conn.commit()
        conn.close()

        return(data, column_names)

    def import_merge_training_data(input_table_1, input_table_2, output_table):
        """
        Import and merge the neural network training data
        """
        start_time = time.time()

        input_1_query = 'SELECT id, value ' + \
                        'FROM ' + input_table_1 + ';'
        input_1_df = pd.DataFrame([list(row) for row in list(_query_database(input_1_query))])
        input_1_df.columns = ['ID', 'Input_Value_1']
        input_1_df.index = input_1_df['ID']

        input_2_query = 'SELECT id, value ' + \
                        'FROM ' + input_table_2 + ';'
        input_2_df = pd.DataFrame([list(row) for row in list(_query_database(input_2_query))])
        input_2_df.columns = ['ID', 'Input_Value_2']
        input_2_df.index = input_2_df['ID']

        output_query = 'SELECT id, value ' + \
                        'FROM ' + output_table + ';'
        output_df = pd.DataFrame([list(row) for row in list(_query_database(output_query))])
        output_df.columns = ['ID', 'Output_Value']
        output_df.index = attribute_df['ID']

        self._input_df = pd.merge(input_1_df, input_2_df.iloc[:, 1:], left_index=True, right_index=True)
        self._input_df = pd.merge(self._input_df, output_df.iloc[:, 1:], left_index=True, right_index=True)
        self._input_df = self._input_df[np.isfinite(self._input_df.iloc[:, -1])]

        self._input_df['Input_Value_1'] = self._input_df['Input_Value_1'].astype('float64')
        self._input_df['Input_Value_1'] = (self._input_df['Input_Value_1'] - np.mean(self._input_df['Input_Value_1'])) \
                                            / np.std(self._input_df['Input_Value_1'])

        print('Importing and merging the training data took {0} seconds'.format(round(time.time() - start_time)))

    def train_model(residues_plot_path):
        """
        Train a neural network on the input data
        """
        train_df, test_df = train_test_split(self._input_df, train_size=0.8, test_size=0.2)

        X = np.array(train_df.iloc[:, :-1]).astype('float64')
        Y = np.array(train_df.iloc[:, -1]).astype('float64')

        self._model = Sequential()
        self._model.add(Dense(100, input_dim=(train_df.shape[1]-1), activation='linear')) # linear, relu
        # self._model.add(Dropout(0.1))
        self._model.add(Dense(1))

        self._model.compile(loss='mean_squared_error', optimizer='adam') # adam, sgd

        self._model.fit(X, Y, epochs=10, batch_size=10)
        mse = self._model.evaluate(X, Y)
        print('The model MSE was: {0}'.format(round(mse, 2)))

        X_pred = np.array(test_df.iloc[:, :-1]).astype('float64')
        Y_pred = np.array(test_df.iloc[:, -1]).astype('float64')

        predictions = [round(preds[0], 3) for preds in self._model.predict(X_pred)]

        plt.rcParams["figure.figsize"] = (16, 9)

        rgba_colors = np.zeros((2, 4))
        rgba_colors[:, 2] = 1.0
        rgba_colors[:, 3] = 0.1

        BOUNDS = [0, 3]
        axes = plt.gca()
        axes.set_xlim(BOUNDS)
        axes.set_ylim(BOUNDS)

        plt.scatter(predictions, Y_pred, color=rgba_colors)
        plt.plot(BOUNDS, BOUNDS, color='green')

        plt.title('Residue Analysis')
        plt.xlabel('Predicted Index')
        plt.ylabel('Actual Index')

        plt.savefig(residues_plot_path)
        plt.clf()

    def import_merge_pred_data(input_path_1, input_path_2):
        """
        Import and merge the neural network prediction data
        """
        start_time = time.time()

        pred_1_df = pd.DataFrame.from_csv(input_path_1, encoding='utf-8', header=0, index_col=None)
        pred_1_df.index = pred_1_df['ID']
        pred_2_df = pd.DataFrame.from_csv(input_path_2, encoding='utf-8', header=0, index_col=None)
        pred_2_df.index = pred_2_df['ID']

        self._pred_df = pd.merge(pred_1_df, pred_2_df.iloc[:, 1:], left_index=True, right_index=True)
        self._pred_df['Input_Value_1'] = self._pred_df['Input_Value_1'].astype('float64')
        self._pred_df['Input_Value_1'] = (self._pred_df['Input_Value_1'] - np.mean(self._pred_df['Input_Value_1'])) \
                                            / np.std(self._pred_df['Input_Value_1'])

        print('Importing and merging the prediction data took {0} seconds'.format(round(time.time() - start_time)))

    def predict_results(prediction_batch_size):
        """
        Use the model to predict attribute values for the max artists
        """
        start_time = time.time()
        processed_rows = 0

        open(self._TEMP_PREDICTIONS_PATH, 'w').close()
        while processed_rows < self._pred_df.shape[0]:
            next_processed_rows = min(processed_rows + prediction_batch_size, self._pred_df.shape[0])
            X_pred = np.array(self._pred_df.iloc[processed_rows:next_processed_rows, :])
            predictions = [round(preds[0], 3) for preds in self._model.predict(X_pred)]

            ids = list(self._pred_df.index[processed_rows:next_processed_rows])

            with open(self._TEMP_PREDICTIONS_PATH, 'a') as f:
                writer = csv.writer(f)
                for i in range(len(predictions)):
                    writer.writerow([ids[i], predictions[i]])

            processed_rows += prediction_batch_size

        print("Predicting the values took {0} seconds".format(round(time.time() - start_time, 2)))

    def export_results(results_table):
        """
        Export predicted attribute values to the appropriate sql table
        """
        start_time = time.time()
        # sys.stderr = open(os.devnull, "w")
        self._query_database('LOAD DATA LOCAL INFILE "' + self._TEMP_PREDICTIONS_PATH + '" ' + \
                                'INTO TABLE ' + results_table + ' ' + \
                                'CHARACTER SET UTF8 ' + \
                                'COLUMNS TERMINATED BY "," ' + \
                                r'ESCAPED BY "\"" ' + \
                                r'ENCLOSED BY "\"" '  + \
                                r'LINES TERMINATED BY "\r\n";')
        print("Inserting the predictions into the results table took {0} seconds".format(round(time.time() - start_time)))
        # sys.stdout = sys.__stderr__
        os.remove(self._TEMP_PREDICTIONS_PATH)


def run(sql_username, sql_password):
    """
    Run the program using the cli inputs
    """
    RANDOM_SEED = 1
    TEMP_PREDICTIONS_PATH = 'temp_predictions.csv'
    SQL_USERNAME = str(sql_username)
    SQL_PASSWORD = str(sql_password)
    DATABASE_IP_ADDRESS = '127.0.0.1'
    DATABASE_PORT_NUMBER = 3306

    INPUT_TABLE_1 = 'input_table_1'
    INPUT_TABLE_2 = 'input_table_2'
    OUTPUT_TABLE = 'output_table'

    RESIDUES_PLOT_PATH = 'residues_plot_path.png'

    INPUT_PATH_1 = 'input_path_1.csv'
    INPUT_PATH_2 = 'input_path_2.csv'

    PREDICTION_BATCH_SIZE = 10000

    RESULTS_TABLE = 'results_table'

    manager = NeuralNetworkManager()
    manager.import_merge_training_data(INPUT_TABLE_1, INPUT_TABLE_2, OUTPUT_TABLE)
    manager.train_model(RESIDUES_PLOT_PATH)
    manager.import_merge_training_data(INPUT_PATH_1, INPUT_PATH_2)
    manager.predict_results(PREDICTION_BATCH_SIZE)
    manager.export_results(RESULTS_TABLE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network Template')
    parser.add_argument('sql_username', help='Username for the sql database')
    parser.add_argument('sql_password', help='Password for the sql database')
    args = parser.parse_args()

    try:
        run(args.sql_username, args.sql_password)
    except Exception as e:
        print(e)
