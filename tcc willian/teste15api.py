import os
import time
import requests
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
import graphviz
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

# Função para iniciar o Apromore via Docker
def rodar_apromore_docker():
    comando = ["docker", "start", "apromore-core"]
    print("Iniciando o Apromore via Docker...")
    processo = subprocess.run(comando, capture_output=True, text=True)
    if processo.returncode == 0:
        print("Apromore iniciado com sucesso.")
    else:
        print("Erro ao iniciar o Apromore:", processo.stderr)

# Função para verificar se o Apromore está disponível
def verificar_apromore_disponivel():
    url = "http://localhost:80"  # URL onde o Apromore deve estar disponível
    while True:
        try:
            resposta = requests.get(url)
            if resposta.status_code == 200:
                print("Apromore está disponível.")
                return True
        except requests.ConnectionError:
            print("Esperando o Apromore ficar disponível...")
            time.sleep(5)

# Função para listar logs disponíveis na pasta
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
    return pd.DataFrame(data)

# Função para adicionar features temporais ao DataFrame
def create_temporal_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    return df

# Função para gerar BPMN via Apromore e renderizar uma imagem
def gerar_bpmn_e_renderizar_imagem_via_apromore(log_path):
    url_upload = "http://localhost:80/apromore-service/upload"  # URL para upload de log no Apromore
    url_bpmn = "http://localhost:80/apromore-service/exportBPMN"  # URL para exportar BPMN

    # Fazendo o upload do log XES para o Apromore
    files = {'file': open(log_path, 'rb')}
    print(f"Enviando o log {log_path} para o Apromore...")
    upload_response = requests.post(url_upload, files=files)

    if upload_response.status_code == 200:
        print("Log enviado com sucesso.")
        # Agora, solicita a geração do modelo BPMN
        print("Solicitando a geração do BPMN...")
        bpmn_response = requests.get(url_bpmn)

        if bpmn_response.status_code == 200:
            # Salvando o BPMN gerado como XML
            bpmn_file = "bpmn_imgs/modelo_bpmn_apromore.xml"
            if not os.path.exists("bpmn_imgs"):
                os.makedirs("bpmn_imgs")
            with open(bpmn_file, 'w') as f:
                f.write(bpmn_response.text)
            print(f"Modelo BPMN gerado e salvo em {bpmn_file}")

            # Agora renderiza o BPMN como imagem (PNG)
            renderizar_bpmn_como_imagem(bpmn_file)
        else:
            print("Erro ao gerar o BPMN:", bpmn_response.status_code)
    else:
        print("Erro ao enviar o log para o Apromore:", upload_response.status_code)

# Função para renderizar o BPMN XML como uma imagem
def renderizar_bpmn_como_imagem(bpmn_file):
    # Aqui você deve implementar como renderizar o BPMN XML para uma imagem
    # Uma abordagem seria usar graphviz, mas isso depende de como o BPMN foi estruturado.
    print(f"Renderizando o BPMN do arquivo {bpmn_file} como uma imagem...")

    # Renderizar com Graphviz (exemplo, você pode precisar adaptar isso)
    dot = graphviz.Digraph(format='png')
    dot.node('Start', shape='circle')
    dot.node('Task1', 'Task 1')
    dot.node('Task2', 'Task 2')
    dot.edge('Start', 'Task1')
    dot.edge('Task1', 'Task2')
    dot.render("bpmn_imgs/modelo_bpmn_apromore")

    print(f"Imagem BPMN gerada e salva em 'bpmn_imgs/modelo_bpmn_apromore.png'.")

# Função para criar o modelo LSTM
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Iniciar o Apromore via Docker e verificar se está disponível
    rodar_apromore_docker()
    verificar_apromore_disponivel()

    # Lista os logs disponíveis
    folder_path = os.path.join(os.path.dirname(__file__), "logs_eventos")
    logs = list_logs_in_folder(folder_path)

    # Seleciona o log desejado
    log_index = int(input("Escolha o número correspondente ao log que deseja utilizar: "))
    log_file = logs[log_index - 1]
    log_path = os.path.join(folder_path, log_file)

    # Gera BPMN via Apromore e renderiza como imagem
    gerar_bpmn_e_renderizar_imagem_via_apromore(log_path)
