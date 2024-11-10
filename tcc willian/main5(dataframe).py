import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pm4py
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.petri_net.visualizer as pn_visualizer

def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs
#Converter Log em DataFrame
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

def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year  
    return df

if __name__ == "__main__":
    # lista os logs
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # mostra os log para ser selecionado
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # carrega o log selecionado
    log = pm4py.read_xes(log_path)

    # descobre uma rede petri usando inductive mine
    net, im, fm = pm4py.discover_petri_net_inductive(log)

    # visualiza e salva a rede Petri 
    gviz_inductive = pn_visualizer.apply(net, im, fm)
    if not os.path.exists("petri_imgs"):
        os.makedirs("petri_imgs")
    pn_visualizer.save(gviz_inductive, "petri_imgs/rede_petri_inductive.png")

    # simula um log a partir da rede petri
    parameters = {"no_traces": 100}
    simulated_log = petri_simulator.apply(net, im, parameters=parameters)

    # salva o log simulado
    if not os.path.exists("simulated_logs"):
        os.makedirs("simulated_logs")
    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log2.xes")
    print("Event log simulado salvo em 'simulated_logs/simulated_log2.xes'")

    # converte o log para um dataframe e salva ele
    simulated_log_df = log_to_dataframe(simulated_log)
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df = create_temporal_features(simulated_log_df)  # Adicionar informações temporais
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log2.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("Event log simulado salvo como DataFrame em 'simulated_logs_dataframes/simulated_log2.csv'")

    # carrega o dataframe p/mineração preditiva
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log2.csv", parse_dates=['timestamp'])

    # os eventos são codificado em inteiro por meio do labelencoder do scikit-learn
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # cria a sequencia de eventos
    sequence_length = 5
    sequences = []
    next_events = []
    #cria pares de sequencia e o evento seguinte
    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        for i in range(len(events) - sequence_length):
            sequences.append(events[i:i + sequence_length])
            next_events.append(events[i + sequence_length])

    # transfoma as lista para arrays de numpy
    X = np.array(sequences)
    y = np.array(next_events)

    # normalização dos dados
    scaler = StandardScaler() #remove a média e escala para a variância unitária
    X = scaler.fit_transform(X)

    # divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # definição dos hiperparametros
    param_grid = {
        'n_estimators': [50, 100, 200], #numero de florestas
        'max_depth': [None, 10, 20, 30], #profundidade max da arvore   
        'min_samples_split': [2, 5, 10],  #numero minimo de amostras para cada no
        'min_samples_leaf': [1, 2, 4]   #numero minimo de amostras para cada no folha
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2) #cv=divisão dos dados de treinamento em partes
    grid_search.fit(X_train, y_train)#busca para encontrar a melhor combinação de hiperparametros

    # melhor estimador
    best_clf = grid_search.best_estimator_

    # Predição nos Dados de Teste
    y_pred = best_clf.predict(X_test)

    # Cálculo da Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")

    # Função de Predição do Próximo Evento
    def predict_next_event(sequence):
        #transforma  os rotulos categoricos em valores numerico pelo laber_encoder para ser usado para predição
        for event in sequence:
            if event not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, event)
        
        sequence_encoded = label_encoder.transform(sequence)
        sequence_encoded = scaler.transform([sequence_encoded[-sequence_length:]])
        predicted_event_encoded = best_clf.predict(sequence_encoded)[0]
        predicted_event = label_encoder.inverse_transform([predicted_event_encoded])[0]
        return predicted_event

    # Exemplo de Uso da Função predict_next_event
    sequence_example = ["start", "activity_A", "activity_B", "activity_C", "activity_D"]
    predicted_next_event = predict_next_event(sequence_example)
    print(f"Próximo evento previsto: {predicted_next_event}")
