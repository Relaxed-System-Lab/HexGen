from request import *
from datetime import datetime
import asyncio

# Modify the IP below before execution
head_node = "http://xxx.xxx.xx.xxx:xxxx"

process_time = []
res_list = []

start = datetime.now()

data = {
    # align with the name specified in worker
    'model_name': 'Llama-2-70b-chat-hf_0',
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
