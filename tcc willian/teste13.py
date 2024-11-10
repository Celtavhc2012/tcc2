import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Função para listar logs disponíveis na pasta
def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

# Converter Log em DataFrame
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

# Criar features temporais a partir dos timestamps
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year  
    return df

# Função para criar o modelo GRU bidirecional com regularização e várias camadas densas
def create_gru_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Camadas GRU Bidirecional
    model.add(Bidirectional(GRU(256, return_sequences=True)))  # Primeira camada GRU bidirecional
    model.add(Dropout(0.4))  
    model.add(Bidirectional(GRU(128)))  # Segunda camada GRU bidirecional
    model.add(Dropout(0.4))

    # Várias camadas densas com regularização L2
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Primeira camada densa
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # Segunda camada densa
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))   # Terceira camada densa
    model.add(Dropout(0.3))

    # Camada de saída
    model.add(Dense(num_classes, activation='softmax'))  # Camada de saída para classificação
    
    # Otimizador Adam
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Lista os logs disponíveis
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Seleciona o log desejado
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Carrega o log selecionado
    log = xes_importer.apply(log_path)

    # Descobre uma rede Petri usando Inductive Miner
    net, im, fm = pm4py.discover_petri_net_inductive(log)

    # Visualiza e salva a rede Petri
    gviz_inductive = pn_visualizer.apply(net, im, fm)
    if not os.path.exists("petri_imgs"):
        os.makedirs("petri_imgs")
    pn_visualizer.save(gviz_inductive, "petri_imgs/rede_petri_inductive.png")

    # Adicionando geração do BPMN usando Inductive Miner
    bpmn_model = pm4py.discover_bpmn_inductive(log)
    gviz_bpmn = bpmn_visualizer.apply(bpmn_model)
    if not os.path.exists("bpmn_imgs"):
        os.makedirs("bpmn_imgs")
    bpmn_visualizer.save(gviz_bpmn, "bpmn_imgs/modelo_bpmn.png")
    print("Modelo BPMN salvo em 'bpmn_imgs/modelo_bpmn.png'")

    # Simula um log a partir da rede Petri
    parameters = {"no_traces": 1000}  
    simulated_log = petri_simulator.apply(net, im, parameters=parameters)

    # Salva o log simulado
    if not os.path.exists("simulated_logs"):
        os.makedirs("simulated_logs")
    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log2.xes")

    # Converte o log para um DataFrame e adiciona features temporais
    simulated_log_df = log_to_dataframe(simulated_log)
    simulated_log_df = create_temporal_features(simulated_log_df)

    # Salva o DataFrame simulado
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log2.csv", index=False)

    # Carrega o DataFrame para mineração preditiva
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log2.csv", parse_dates=['timestamp'])

    # Codificação de eventos para números inteiros com LabelEncoder
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # Aumentar o tamanho da sequência para capturar mais contexto temporal
    sequence_length = 3  # Reduzi para capturar mais dados com logs menores

    sequences = []
    next_events = []

    # Verificar se há eventos suficientes para gerar sequências
    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        print(f"Caso {caseid} tem {len(events)} eventos.")  # Debugging
        if len(events) >= sequence_length:
            for i in range(len(events) - sequence_length):
                sequences.append(events[i:i + sequence_length])
                next_events.append(events[i + sequence_length])

    X = np.array(sequences)
    y = np.array(next_events)

    # Verificar se o array X está vazio
    if X.size == 0:
        print("O array X está vazio. Verifique os dados de entrada.")
        exit()

    # Definir o scaler antes de usar
    scaler = StandardScaler()

    # Normalização dos dados
    X = scaler.fit_transform(X)

    # Verificar número correto de classes
    num_classes = len(np.unique(y))
    print(f"Número de classes: {num_classes}")

    if np.max(y) >= num_classes:
        y = np.clip(y, 0, num_classes - 1)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape dos dados para a GRU
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Criação e treinamento do modelo GRU com callbacks
    gru_model = create_gru_model((X_train.shape[1], 1), num_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    # Avaliação do modelo
    loss, accuracy = gru_model.evaluate(X_test, y_test)
    print(f"Acurácia: {accuracy:.2f}")
