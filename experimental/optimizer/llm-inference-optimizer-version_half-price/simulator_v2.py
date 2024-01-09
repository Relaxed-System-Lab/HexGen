import asyncio
from functools import partial
import unittest

import ray

from alpa_serve.profiling import ParallelConfig, load_test_prof_result
from alpa_serve.controller import run_controller
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, Request, PoissonProcess, DeterministicProcess, GammaProcess
from cost_model_impl import compute_costs, tp_communication_costs

import numpy as np

# plan: [[8, 4, 4], [8, 4, 4], [8, 4, 4], [8, 4, 4], [8, 4, 4], [16, 8, 8], [8, 4, 4]]

class Simulator:
    
    def __init__(self, plans, bias, bsz, slo):
        self.plans = plans
        self.bias = bias
        self.bsz = bsz
        self.slo = slo

    def calculate_stage_costs(self, plan, bias, bsz):
        stage_costs, _, pp_layer_list = compute_costs(plan, bias, bsz)
        stage_costs_tp = list(np.array(stage_costs) / np.array(plan))
        comm_costs = tp_communication_costs(pp_layer_list, plan, bsz)
        stage_costs = [x + y for x, y in zip(stage_costs_tp, comm_costs)]
        return list(np.array(stage_costs) * 0.001)

    def register_models(self, controller):
        for i in range(len(self.plans)):
            stage_costs = self.calculate_stage_costs(self.plans[i], self.bias[0], self.bsz[0])
            stage_costs = len(stage_costs) * [(0.010137)]
            controller.register_model.remote(
                    f"model_{i}", 
                    partial(
                        Executable, 
                        load_test_prof_result(
                            "Llama-70b", # Select model type
                            stage_costs, # Input model plan
                            ),
                        ),
                    )
            group_id = i
            controller.create_mesh_group_manager.remote(
                    group_id, 
                    [1, len(self.plans[i])],
                    )
            controller.create_replica.remote(
                    f"model_{i}", 
                    group_id, 
                    [ParallelConfig(1, 1, len(self.plans[i]))],
                    )
    
    
    def gen_workload(self, slo):
        """Generate a training workload for search."""
        ws = []
        for i in range(len(self.plans)):
            ws.append(PoissonProcess(200).generate_workload(
                len(self.plans), f"model_{i}", 0, duration=20,
                slo=slo))
        train_workload = Workload.merge(*ws)
        return train_workload

    async def event(self):
        # Define Client and register models
        controller = Controller()
        self.register_models(controller)
        client = Client(controller)
        
        # Define workloads (currently PoissonProcess)
        w = self.gen_workload(self.slo)

        # Submit workloads
        await client.submit_workload(w)

        return client, w
       
    def exec(self):
        client, w = run_event_loop(self.event())
        stats = client.compute_stats(w, warmup=1)
        # Workload.print_stats(stats)
        return stats.goodput


# # Example to implement the Simulator class
# slo = 1
# plans = [[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1]]
# plans = [[8, 4, 4], [8, 4, 4], [8, 4, 4], [8, 4, 4], [8, 4, 4], [16, 8, 8], [8, 4, 4]]
# simulator = Simulator(plans, slo)
# goodput = simulator.exec()
# print(goodput)
