import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pm4py
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Função para listar os arquivos de log na pasta
def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

# Função para converter o log em DataFrame
def log_to_dataframe(log):
    data = []
    for trace in log:
        caseid = trace.attributes.get("concept:name", None)
        if caseid is None:
            continue
        for event in trace:
            event_data = {
                "caseid": caseid,
                "event": event.get("concept:name", "unknown"),
                "timestamp": event.get("time:timestamp", None)
            }
            data.append(event_data)
    
    df = pd.DataFrame(data)
    return df

# Adicionar características temporais ao DataFrame
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year  
    return df

# Função para criar o modelo LSTM
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output para as classes de eventos
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Lista os logs
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Mostra os logs para serem selecionados
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Carrega o log selecionado com tratamento de erros
    try:
        log = pm4py.read_xes(log_path)
        print(f"Log '{log_file}' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o log: {e}")
        exit(1)

    # Descobre o modelo BPMN usando o minerador indutivo
    bpmn_model = pm4py.discover_bpmn_inductive(log)

    # Visualiza e salva o modelo BPMN
    gviz = bpmn_visualizer.apply(bpmn_model)
    if not os.path.exists("bpmn_imgs"):
        os.makedirs("bpmn_imgs")
    bpmn_visualizer.save(gviz, "bpmn_imgs/modelo_bpmn.png")
    print("Imagem BPMN salva em 'bpmn_imgs/modelo_bpmn.png'")
    bpmn_visualizer.view(gviz)  # Exibe a imagem BPMN

    # Converte o log para um DataFrame e o salva
    simulated_log_df = log_to_dataframe(log)
    if not os.path.exists("simulated_logs_dataframes"):
        os.makedirs("simulated_logs_dataframes")
    simulated_log_df = create_temporal_features(simulated_log_df)  # Adicionar informações temporais
    simulated_log_df.to_csv("simulated_logs_dataframes/simulated_log.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
    print("Event log salvo como DataFrame em 'simulated_logs_dataframes/simulated_log.csv'")

    # Carrega o DataFrame para mineração preditiva
    simulated_log_df = pd.read_csv("simulated_logs_dataframes/simulated_log.csv", parse_dates=['timestamp'])

    # Eventos são codificados como inteiros usando o LabelEncoder do scikit-learn
    label_encoder = LabelEncoder()
    simulated_log_df['event'] = label_encoder.fit_transform(simulated_log_df['event'])

    # Exemplo de uso: criação das sequências de eventos
    sequence_length = 5
    sequences = []
    next_events = []

    # Cria pares de sequência e o evento seguinte
    for caseid, group in simulated_log_df.groupby('caseid'):
        events = group['event'].values
        for i in range(len(events) - sequence_length):
            sequences.append(events[i:i + sequence_length])
            next_events.append(events[i + sequence_length])

    # Transforma as listas em arrays numpy
    X = np.array(sequences)
    y = np.array(next_events)

    # Normalização dos dados
    scaler = StandardScaler()  # Remove a média e escala para a variância unitária
    X = scaler.fit_transform(X)

    # Transforma as saídas em formato one-hot encoding
    y = to_categorical(y)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Redimensiona X para que seja aceito pelo LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Cria o modelo LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])  # time steps, features
    num_classes = y_train.shape[1]  # número de eventos distintos
    model = create_lstm_model(input_shape, num_classes)

    # Treina o modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Avalia o modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

    # Função para predizer o próximo evento
    def predict_next_event(sequence):
        sequence_encoded = label_encoder.transform(sequence)
        sequence_encoded = scaler.transform([sequence_encoded[-sequence_length:]])
        sequence_encoded = sequence_encoded.reshape((1, sequence_length, 1))
        predicted_event_encoded = model.predict(sequence_encoded)
        predicted_event = label_encoder.inverse_transform([np.argmax(predicted_event_encoded)])[0]
        return predicted_event

    # Exemplo de Uso da Função predict_next_event
    sequence_example = ["start", "activity_A", "activity_B", "activity_C", "activity_D"]
    predicted_next_event = predict_next_event(sequence_example)
    print(f"Próximo evento previsto: {predicted_next_event}")
