# Python Prototypes

## Description:
This repo provides a Dockerized prototyping environment for testing python notebooks and scripts.  A sample of example and template notebooks and scripts has been included, which mostly involve modeling with pandas and scikit-learn.

## Usage:
1. After initial repo clone, setup the repo as follows ` chmod +x ./bin/setup_repo.sh && ./bin/setup_repo.sh `
2. Build the dockerized Jupyter environment ` make build `
3. Start the notebook Jupyter server ` make upn `
4. View and edit notebooks at http://localhost:8888
5. Stop the Jupyter server ` make stop `
6. Lint the python scripts ` make flake `
7. Run the script in the environment ` ./bin/run_script.sh scripts/examples/SCRIPT_NAME.py ARGS `

## Content

#### Notebook Examples:
* **Machine Learning Pipeline:** *notebooks/examples/Machine_Learning_Pipeline.ipynb*
  * Build a comprehensive machine learning pipeline template using scikit-learn

#### Script Examples:
* **Classfier Pickling:** *scripts/examples/classifier_pickling.py*
  * Builds a classifier on sklearn wine input data using the desired model type, along with pickling
  * Usage: ` python3 scripts/examples/classifier_pickling.py MODEL_STR OUTPUT_DIR `
  * Example: ` ./bin/run_script.sh scripts/examples/classifier_pickling.py rf output/temp `
* **JSON to DataFrame Parser:** *scripts/examples/json_to_dataframe_parser.py*
  * Converts a generic json with field-path info into a dataframe
  * Usage: ` python3 json_to_dataframe_parser.py JSON_INPUT_PATH FIELD_INFO_PATH DATAFRAME_OUTPUT_PATH `
  * Example: ` ./bin/run_script.sh scripts/examples/json_to_dataframe_parser.py scripts/data/sample.json scripts/data/field_info.csv output/temp.csv `
* **K-Means Clustering:** *scripts/examples/k_means_clustering_example.py*
  * Runs k-means clustering on sklearn iris input data and outputs the appropriate visualization
  * Usage: ` python3 scripts/examples/k_means_clustering_example.py OUTPUT_DIR `
  * Example: ` ./bin/run_script.sh scripts/examples/k_means_clustering_example.py output/temp `
* **Machine Learning Pipeline:** *scripts/examples/machine_learning_pipeline.py*
  * Builds and runs a standard machine learning pipeline
  * Usage: ` python3 scripts/examples/machine_learning_pipeline.py OUTPUT_DIR SHOULD_SHOW_OUTPUT `
  * Example: ` ./bin/run_script.sh scripts/examples/machine_learning_pipeline.py output/temp -v `
* **ROC Plotting:** *scripts/examples/roc_plotting.py*
  * Builds a receiver operating characteristic curve for sklearn breast cancer input data using the desired model type
  * Usage: ` python3 scripts/examples/roc_plotting.py MODEL_STR OUTPUT_DIR `
  * Example: ` ./bin/run_script.sh scripts/examples/roc_plotting.py rf output/temp `
* **XML to JSON Parser:** *scripts/examples/xml_to_json_parser.py*
  * Converts a generic xml file into json
  * Usage: ` python3 xml_to_json_parser.py XML_INPUT_PATH JSON_OUTPUT_PATH `
  * Example: ` ./bin/run_script.sh scripts/examples/xml_to_json_parser.py scripts/data/sample.xml output/temp.json `

#### Script Templates:
* **Neural Network Template:** *scripts/templates/neural_network_pipeline.py*
  * Template for building, training and predicting with a neural network using the Keras API
