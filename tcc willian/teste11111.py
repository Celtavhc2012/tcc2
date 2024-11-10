import pm4py
import os
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.petri_net.visualizer as pn_visualizer
import pandas as pd
import numpy as np
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

def log_to_dataframe(log):
    data = []
    for trace in log:
        caseid = trace.attributes["concept:name"]
        for event in trace:
            event_data = {
                "caseid": caseid,
                "event": event["concept:name"],
                "timestamp": event["time:timestamp"]
            }
            data.append(event_data)
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # List logs in the folder
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Select log file
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Read the selected log file
    log = pm4py.read_xes(log_path)

    # Discover Petri net using Inductive Miner
    net, im, fm = pm4py.discover_petri_net_inductive(log)

    # Visualize and save the discovered Petri net
    gviz_inductive = pn_visualizer.apply(net, im, fm)
    if not os.path.exists("petri_imgs"):
        os.makedirs("petri_imgs")
    pn_visualizer.save(gviz_inductive, "petri_imgs/rede_petri_inductive.png")

    # Simulate event log from Petri net
    parameters = {"no_traces": 100}
    simulated_log = petri_simulator.apply(net, im, parameters=parameters)

    # Save simulated log to XES file
    if not os.path.exists("simulated_logs"):
        os.makedirs("simulated_logs")
    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log2.xes")
    print("Event log simulado salvo em 'simulated_logs/simulated_log2.xes'")

    # Convert simulated log to DataFrame and save as CSV
    simulated_log_df = log_to_dataframe(simulated_log)
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log2.csv", index=False)
    print("Event log simulado salvo como DataFrame em 'simulated_logs_dataframes/simulated_log2.csv'")

    # Load the DataFrame for predictive process mining
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log2.csv")

    # Encode events as integers
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # Create sequences of events (sliding window)
    sequence_length = 5  # Define the sequence length
    sequences = []
    next_events = []

    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        for i in range(len(events) - sequence_length):
            sequences.append(events[i:i + sequence_length])
            next_events.append(events[i + sequence_length])

    # Transform lists to numpy arrays
    X = np.array(sequences)
    y = np.array(next_events)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")

    # Function to predict the next event
    def predict_next_event(sequence):
        # Ensure all events in the sequence are in the label encoder's classes
        for event in sequence:
            if event not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, event)
        
        sequence_encoded = label_encoder.transform(sequence)
        predicted_event_encoded = clf.predict([sequence_encoded[-sequence_length:]])[0]
        predicted_event = label_encoder.inverse_transform([predicted_event_encoded])[0]
        return predicted_event

    # Example usage of the prediction function
    sequence_example = ["start", "activity_A", "activity_B", "activity_C", "activity_D"]
    predicted_next_event = predict_next_event(sequence_example)
    print(f"Próximo evento previsto: {predicted_next_event}")
