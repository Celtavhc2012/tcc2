import os
import pm4py
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

# Função para gerar BPMN diretamente do log de eventos
def gerar_bpmn_com_pm4py(log_path):
    # Carregar o log XES
    log = pm4py.read_xes(log_path)

    # Descobrir o modelo BPMN a partir do log de eventos
    bpmn_model = pm4py.discover_bpmn_inductive(log)

    # Visualizar e exportar o diagrama BPMN como imagem
    gviz = bpmn_visualizer.apply(bpmn_model)
    
    # Verifica e cria o diretório de imagens BPMN
    if not os.path.exists("bpmn_imgs"):
        os.makedirs("bpmn_imgs")
    
    # Salvar a imagem BPMN gerada
    bpmn_visualizer.save(gviz, "bpmn_imgs/modelo_bpmn_pm4py.png")
    print("Imagem BPMN gerada e salva em 'bpmn_imgs/modelo_bpmn_pm4py.png'.")

# Exemplo de chamada da função
log_path = "caminho/para/o/seu_log.xes"  # Insira o caminho correto para o arquivo de log XES
gerar_bpmn_com_pm4py(log_path)
