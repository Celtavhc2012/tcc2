import pm4py
import os
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.petri_net.visualizer as pn_visualizer
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

def prepare_data_for_ppm(log):
    data = pm4py.convert_to_dataframe(log)
    data = dataframe_utils.convert_timestamp_columns_in_df(data)
    data = pm4py.filtering.log_attributes.attributes_filter.apply(data, parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "concept:name"})

    # Create sequences of events
    sequences = data.groupby("case:concept:name")["concept:name"].apply(list).values
    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])
    
    # Encode the event labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = [le.transform(x) for x in X]

    # Padding sequences to have the same length
    max_len = max(len(seq) for seq in X)
    X = [seq + [0]*(max_len-len(seq)) for seq in X]

    return train_test_split(X, y, test_size=0.2, random_state=42), le

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    log = pm4py.read_xes(log_path)

    net, im, fm = pm4py.discover_petri_net_inductive(log)

    print(f'Petri net descoberta pelo Inductive Miner:')

    gviz_inductive = pn_visualizer.apply(net, im, fm)
    if not os.path.exists("petri_imgs"):
        os.makedirs("petri_imgs")
    pn_visualizer.save(gviz_inductive, "petri_imgs/rede_petri_inductive.png")

    # Simulate event log from Petri net
    parameters = {"no_traces": 100}
    simulated_log = petri_simulator.apply(net, im, parameters=parameters)

    if not os.path.exists("simulated_logs"):
        os.makedirs("simulated_logs")

    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log.xes")
    print("Event log simulado salvo em 'simulated_logs/simulated_log.xes'")

    # Prepare data for PPM
    (X_train, X_test, y_train, y_test), le = prepare_data_for_ppm(simulated_log)

    # Train a predictive model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the predictive model: {accuracy * 100:.2f}%')
