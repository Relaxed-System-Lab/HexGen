## Send Request 

Once you have started service on head coordinator and worker coordinators, you could send request to them. 

First modify the input in `single_reques.py`, just make sure you add an suffix `_0` to correctly call the rank-0, an example is
```python
data = {
    'model_name': 'Llama-2-70b-chat-hf_0',
    'params': {
        'prompt': "Do you like your self? ",
        'max_new_tokens': 128,
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 40,
    }
}
``` 


By running the following command, you will see the answer to prompt, pure inference time and over time, in a python tuple.

```python
python3 single_request.py
```

functions in `request.py` provides for retrieving answers and checking nodes' status, respectively.
