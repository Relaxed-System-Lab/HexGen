import aiohttp
import time

async def request_head_node(data, head_node, task_id=0):
    start = time.time()
    
    # this large timeout is useful when tasks are crowded on coordinato
    timeout = aiohttp.ClientTimeout(total=60 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        endpoint = f"{head_node}/api/v1/request/inference"
        resp = await session.post(endpoint, json=data)
        result = await resp.json()
        if 'error' in result:
            return None, None, None

        prompt_resp, infer_time = eval(result['data'])
        print(f"##### task {task_id} has finished inference #####")


        return prompt_resp, infer_time, time.time() - start

async def check_status(head_node):
    async with aiohttp.ClientSession() as session:
        endpoint = f"{head_node}/api/v1/status/peers"
        resp = await session.get(endpoint)
        result = await resp.json()
        return result



