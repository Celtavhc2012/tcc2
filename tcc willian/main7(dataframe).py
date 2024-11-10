# Importando bibliotecas necessárias
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import pm4py
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.petri_net.visualizer as pn_visualizer

# Função para listar arquivos.xes em um diretório
def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

# Função para converter log para DataFrame
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

# Função para criar recursos temporais
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    return df

# Função para pré-processar dados
def preprocess_data(simulated_log_df, sequence_length):
    # Codificar eventos como inteiros
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # Criar sequências de eventos (janela deslizante)
    sequences = []
    next_events = []

    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        if len(events) > sequence_length:  # Garantir que há eventos suficientes para criar uma sequência
            for i in range(len(events) - sequence_length):
                sequences.append(events[i:i + sequence_length])
                next_events.append(events[i + sequence_length])

    # Transformar listas em arrays numpy
    X = np.array(sequences)
    y = np.array(next_events)

    # Retornar dados pré-processados e codificador de rótulos para uso posterior
    return X, y, label_encoder

# Execução principal
if __name__ == "__main__":
    # Listar logs no diretório
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Selecionar arquivo de log
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Ler o arquivo de log selecionado
    log = pm4py.read_xes(log_path)

    # Descobrir Petri net usando Inductive Miner
    net, im, fm = pm4py.discover_petri_net_inductive(log)

    # Visualizar e salvar o Petri net descoberto
    gviz_inductive = pn_visualizer.apply(net, im, fm)
    if not os.path.exists("petri_imgs"):
        os.makedirs("petri_imgs")
    pn_visualizer.save(gviz_inductive, "petri_imgs/rede_petri_inductive.png")

    # Simular log de eventos a partir do Petri net
    parameters = {"no_traces": 100}
    simulated_log = petri_simulator.apply(net, im, parameters=parameters)

    # Salvar log simulado em arquivo XES
    if not os.path.exists("simulated_logs"):
        os.makedirs("simulated_logs")
    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log2.xes")
    print("Event log simulado salvo em 'simulated_logs/simulated_log2.xes'")

    # Converter log simulado para DataFrame e salvar como CSV
    simulated_log_df = log_to_dataframe(simulated_log)
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df = create_temporal_features(simulated_log_df)  # Adicionar informações temporais
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log2.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("Event log simulado salvo como DataFrame em 'simulated_logs_dataframes/simulated_log2.csv'")

    # Carregar o DataFrame para mineração preditiva de processos
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log2.csv", parse_dates=['timestamp'])

    # Pré-processar os dados
    sequence_length = 3  # Definir o comprimento da sequência
    X, y, label_encoder = preprocess_data(simulated_log_df, sequence_length)

    # Verificar se as sequências foram criadas corretamente
    if X.size == 0:
        raise ValueError("O array X está vazio. Verifique a etapa de criação das sequências para problemas.")

    # Padronizar recursos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE para balancear o conjunto de dados
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Ajuste de hiperparâmetros usando Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2) 
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Melhor estimador
    best_clf = grid_search.best_estimator_

    # Fazer previsões no conjunto de teste
    y_pred = best_clf.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")

    # Função para prever o próximo evento
    def predict_next_event(sequence):
        # Garantir que todos os eventos na sequência estão nas classes do codificador de rótulos
        for event in sequence:
            if event not in label_encoder.classes_:
                label_encoder.classes_ = np.append(label_encoder.classes_, event)
        
        sequence_encoded = label_encoder.transform(sequence)
        sequence_encoded = scaler.transform([sequence_encoded[-sequence_length:]])
        predicted_event_encoded = best_clf.predict(sequence_encoded)[0]
        predicted_event = label_encoder.inverse_transform([predicted_event_encoded])[0]
        return predicted_event

    # Exemplo de uso da função de previsão
    sequence_example = ["start", "activity_A", "activity_B"]
    predicted_next_event = predict_next_event(sequence_example)
    print(f"Próximo evento previsto: {predicted_next_event}")
