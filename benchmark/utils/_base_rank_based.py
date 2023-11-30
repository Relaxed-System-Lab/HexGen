import json
import signal
import asyncio
import os
import re
from loguru import logger
from nats.aio.client import Client as NATS
import aiohttp
import torch.distributed as dist
import torch
from threading import Thread
from multiprocessing import Process
from _utils import get_visible_gpus_specs

async def shutdown(signal, loop, nc, model_name, connection_notice):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Gracefully shutting down {model_name} worker...")

    tasks = [t for t in asyncio.all_tasks() if t is not
             asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks)
    connection_notice['status'] = 'disconnected'
    await nc.publish("worker:status", bytes(f"{json.dumps(connection_notice)}", encoding='utf-8'))
    await nc.close()
    loop.stop()

class InferenceWorker():
    def __init__(self, model_name, head_node, group_rank, global_rank, priority=None, args=None) -> None:

        self.addr = self.set_addr()
        
        self.group_rank = group_rank
        self.global_rank = global_rank
        self.world_size = args.world_size

        self.model_name = model_name

        self.nc = NATS()
        self.connection_notice = {}
        
        self.head_node = head_node
        self.occupied_ids = []
        
        self.priority = priority

        # reserve for future design
        # group_id = asyncio.run(self.generate_group_id())

        self.group_id = args.group_id
        print(f"group id is {self.group_id}")

    async def run(self, loop):
        await self.nc.connect(f"nats://{self.addr}:8094")    
        print(f"nats://{self.addr}:8094")
        if self.group_rank == 0:
            name = f"inference:{self.model_name}_{self.group_rank}"
        else:
            name = f"inference:{self.model_name}_{self.group_rank}_{self.group_id}"
        
        # reserve for future design
        if self.group_rank == 0:    
            await self.nc.subscribe(name, "workers", self.process_request)
        else:
            await self.nc.subscribe(name, "workers", self.process_request)

        self.connection_notice = {
            'service': name,
            'gpus': get_visible_gpus_specs(),
            'client_id': self.nc.client_id,
            'status': 'connected',
            'group_rank': self.group_rank,
            'global_rank': self.global_rank,
            'group_id': int(self.group_id),
            }
        
        await self.nc.publish("worker:status", bytes(f"{json.dumps(self.connection_notice)}", encoding='utf-8'))

    async def process_request(self, msg):
        processed_msg = json.loads(msg.data.decode())
        result = await self.handle_requests(processed_msg['params'])

        await self.reply(msg, result)

    async def handle_requests(self, msg):
        raise NotImplementedError

    async def reply(self, msg, data):
        data = json.dumps(data)
        await self.nc.publish(msg.reply, bytes(data, encoding='utf-8'))
    
    async def launch(self, params, rank):

        print(f"send request is calling {rank} on group {self.group_id}")
        async with aiohttp.ClientSession() as session:
            endpoint = f"http://{self.addr}:8092/api/v1/request/_inference"
            new_msg = {}
            if rank != 0:
                new_msg['model_name'] = f'{self.model_name}_{rank}_{self.group_id}'
            else:
                if self.priority is not None:
                    new_msg['model_name'] = f'{self.model_name}_{rank}_{self.priority}'
            new_msg['params'] = params
            resp = await session.post(endpoint, json=new_msg)

            return resp
    
    def send_request(self, params, rank):

        res = asyncio.run(self.launch(params, rank))

    def start(self):
        logger.info(f"Starting {self.model_name} worker on rank {self.group_rank}...")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGABRT, signal.SIGTSTP)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s, loop, self.nc, self.model_name, self.connection_notice)))
        loop.run_until_complete(self.run(loop))
        loop.run_forever()
        loop.close()

    def set_addr(self):
        try:         
            addr = os.environ['MASTER_ADDR']
        except KeyError:
            addr = 'localhost'

        return addr
    
    def set_group_id(self, group_id):
        self.model_name = self.model_name + f"_{group_id}"
    
    # auto assignment for group id, reserve for future design
    async def generate_group_id(self):
        async with aiohttp.ClientSession() as session:
            used_group_ids = []
            started_ranks = 0
            endpoint = f"{self.head_node}/api/v1/status/peers"  
            resp = await session.get(endpoint)
            connection_infos = await resp.json()

            for peer in connection_infos['peers']:
                if peer['service'] is not None:
                    for service in peer['service']:
                        service_suffix = re.findall(r"_\d+_\d+", service['name'])
                        if service_suffix:
                            service_suffix = re.findall(r"\d+", service_suffix[0])
                            used_group_ids.append(int(service_suffix[-1]))
                    started_ranks = len(used_group_ids)
                    print(used_group_ids)
                    used_group_ids = sorted(list(set(used_group_ids)))
            if len(used_group_ids) == 0:
                return 0
            else:
                return max(used_group_ids) + 1 if self.world_size == started_ranks else max(used_group_ids)
    
    def parse_msg(self, msg):
        model_msg = {}
        
        model_msg['prompt'] = msg.get('prompt', '')
        model_msg['max_new_tokens'] = msg.get('max_new_tokens', 128)
        model_msg['temperature'] = msg.get('temperature', 0.9)
        model_msg['top_k'] = msg.get('top_k', 50)
        model_msg['top_p'] = msg.get('top_p', 0.9)

        return model_msg
