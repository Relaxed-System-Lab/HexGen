from request import *
from datetime import datetime
import asyncio

head_node = "http://198.176.96.165:8092"

process_time = []
res_list = []

start = datetime.now()

data = {
    'model_name': 'Llama-2-7b-chat-hf_0',
    'params': {
        'prompt': "Do you like your self? ",
        'max_new_tokens': 128,
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 40,
    }
}

res = asyncio.run(request_head_node(data, head_node=head_node))

end = datetime.now()

res_list.append(res)
process_time.append(end - start)    # each element is a timedelta

print(res)
print(process_time)

print("=" * 40)

status = asyncio.run(check_status(head_node))
print(status)
