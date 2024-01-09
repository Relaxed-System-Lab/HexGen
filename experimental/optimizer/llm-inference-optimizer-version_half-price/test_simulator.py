from simulator_interface import PlacementSimulator

simulator = PlacementSimulator([[1,1,1,1],[1,1,1,1],[2,2,2,2]], [1,1,1], "Llama_70B")
latency = simulator.run_simulation()
print(latency)