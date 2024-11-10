import pm4py
import os
from pm4py.algo.simulation.playout.petri_net import algorithm as petri_simulator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py.visualization.petri_net.visualizer as pn_visualizer

def list_logs_in_folder(folder_path):
    print("Logs disponíveis na pasta:")
    logs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xes')]
    for i, log_file in enumerate(logs):
        print(f"{i+1}. {log_file}")
    return logs

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

    xes_exporter.apply(simulated_log, "simulated_logs/simulated_log2.xes")

    print("Event log simulado salvo em 'simulated_logs/simulated_log.xes'")
