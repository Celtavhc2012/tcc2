import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.bpmn.visualizer as bpmn_visualizer  # Import para visualização do BPMN

# Função para listar logs na pasta
def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

# Função para converter log em DataFrame
def log_to_dataframe(log):
    data = []
    # Verificando a estrutura de cada trace
    for trace in log:
        if "concept:name" in trace.attributes:  # Verifica se o trace tem o atributo esperado
            caseid = trace.attributes["concept:name"]  # Nome do caso
        else:
            caseid = f"Case-{log.index(trace)}"  # Se não tiver, cria um identificador para o caso
        
        for event in trace:
            if "concept:name" in event and "time:timestamp" in event:  # Verifica se os atributos do evento existem
                event_data = {
                    "caseid": caseid,  # Atributo "caseid"
                    "event": event["concept:name"],  # Nome do evento
                    "timestamp": event["time:timestamp"]  # Timestamp do evento
                }
                data.append(event_data)
            else:
                print(f"Evento ignorado no trace {caseid}, pois está faltando 'concept:name' ou 'time:timestamp'")
    df = pd.DataFrame(data)
    return df

# Função para criar variáveis temporais
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year  
    return df

if __name__ == "__main__":
    # Lista os logs
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Mostra os logs para ser selecionado
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Carrega o log selecionado
    log = pm4py.read_xes(log_path)

    # Descobre o modelo BPMN usando Inductive Miner
    bpmn_model = pm4py.discover_bpmn_inductive(log)

    # Visualiza e salva o modelo BPMN
    gviz_bpmn = bpmn_visualizer.apply(bpmn_model)
    if not os.path.exists("bpmn_imgs"):
        os.makedirs("bpmn_imgs")
    bpmn_visualizer.save(gviz_bpmn, "bpmn_imgs/modelo_bpmn.png")
    print("Modelo BPMN salvo em 'bpmn_imgs/modelo_bpmn.png'")

    # Converte o log para um DataFrame e salva ele
    simulated_log_df = log_to_dataframe(log)
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df = create_temporal_features(simulated_log_df)  # Adicionar informações temporais
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log_bpmn.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("Event log salvo como DataFrame em 'simulated_logs_dataframes/simulated_log_bpmn.csv'")

    # Carrega o DataFrame para mineração preditiva
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log_bpmn.csv", parse_dates=['timestamp'])

    # Os eventos são codificados como inteiros pelo LabelEncoder do scikit-learn
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # Criação da sequência de eventos
    sequence_length = 5
    sequences = []
    next_events = []
    
    # Cria pares de sequência e o evento seguinte
    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        for i in range(len(events) - sequence_length):
            sequences.append(events[i:i + sequence_length])
            next_events.append(events[i + sequence_length])

    # Transforma as listas em arrays do numpy
    X = np.array(sequences)
    y = np.array(next_events)

    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definição dos hiperparâmetros para o RandomForest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Criação do modelo RandomForest
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_clf = grid_search.best_estimator_

    # Predição nos dados de teste
    y_pred = best_clf.predict(X_test)

    # Cálculo da acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")

    # Função de predição do próximo evento
    def predict_next_event(sequence):
        # Verifica se os eventos estão no LabelEncoder
        for event in sequence:
            if event not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, event)
        
        # Transforma a sequência em valores codificados
        sequence_encoded = label_encoder.transform(sequence)
        sequence_encoded = scaler.transform([sequence_encoded[-sequence_length:]])
        
        # Prediz o próximo evento
        predicted_event_encoded = best_clf.predict(sequence_encoded)[0]
        predicted_event = label_encoder.inverse_transform([predicted_event_encoded])[0]
        return predicted_event

    # Exemplo de uso da função predict_next_event
    sequence_example = ["start", "activity_A", "activity_B", "activity_C", "activity_D"]
    predicted_next_event = predict_next_event(sequence_example)
    print(f"Próximo evento previsto: {predicted_next_event}")
