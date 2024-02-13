from llmsim.cluster import Cluster, load_meshexecutors
from llmsim.scheduler import FIFOScheduler
from llmsim.workload import PossoinWorkLoad
from llmsim.simulator import Simulator
from llmsim.utils import compute_statistics_from_cluster_trace, compute_statistics_from_simulation, \
                          dump_chrome_tracing_from_simulation, dump_chrome_tracing_from_cluster_trace
from placements.gen_placement_json import generate_cases, create_json
from llmsim.model import generate_model_configs
from llmsim.workload import generate_workload 
from gen_model_id_to_service_name import gen_model_id_to_service_name

class PlacementSimulator:
    def __init__(self, device_counts_array, num_replicas, model_name, num_nodes=1, memory_capacity=16):
        self.device_counts_array = device_counts_array
        self.num_replicas = num_replicas
        self.num_nodes = num_nodes
        self.memory_capacity = memory_capacity
        self.model_name = model_name
        self.total_devices = sum(sum(sublist) for sublist in device_counts_array)
        self.num_model = len(device_counts_array)
        self.total_num_models = sum(num_replicas)

    def run_simulation(self):
        cases = generate_cases(self.total_devices, self.model_name, self.device_counts_array)
        create_json(cases, num_replicas=self.num_replicas)
        generate_workload(self.total_num_models, 10, [1/self.total_num_models] * self.total_num_models, 20, [0.01] * self.total_num_models, f"./workload/interface_test_{self.total_num_models}_models")
        model_configs = generate_model_configs(self.device_counts_array)
        cluster = Cluster(self.num_nodes, self.total_devices, self.memory_capacity)
        meshexecutors = load_meshexecutors(f"./placements/interface_test.json", cluster, model_configs)
        workload = PossoinWorkLoad.load(f"./workload/interface_test_{self.total_num_models}_models")
        model_id_to_service_name = gen_model_id_to_service_name(self.num_model, self.num_replicas)
        scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
        simulator = Simulator(scheduler, cluster)
        simulator.start()
        _, latency = compute_statistics_from_simulation(scheduler.completed_tasks)
        return latency

