import os
import openai
from tenacity import (
    retry,
    wait_random_exponential,
    stop_never
)  # for exponential backoff
from openai import OpenAI

proxy_url = "http://localhost:7890"
api_base = "http://localhost:8000/v1"

def set_proxy(url=proxy_url):
    if url:
        os.environ["http_proxy"] = url
        os.environ["https_proxy"] = url
        os.environ["HTTP_PROXY"] = url
        os.environ["HTTPS_PROXY"] = url

def unset_proxy():
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)


def get_output(res):
    out = res['message']['content']
    if res['finish_reason'] != 'stop':
        out += '<|NONSTOP|>'
    return out

def get_output_raw(res):
    out = res['text']
    if res['finish_reason'] != 'stop':
        out += '<|NONSTOP|>'
    return out
       

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_never)
def decoder_for_openai(model_name, input, max_tokens, temperature=0.7, top_p=0.95, apikey=None, n=1, stop=None, sys_prompt=None):
    frequency_penalty = 0
    presence_penalty = 0
    if sys_prompt:
        sys_prompt_content = sys_prompt
    else:
        sys_prompt_content = "You are a helpful assistant."

    # print(apikey)
    # print('start')
    # openai.api_key = apikey
    client = OpenAI(base_url = 'http://localhost:11434/v1',api_key=apikey)

    
    set_proxy(proxy_url)
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        # api_key=apikey,
        model="llama2",
        messages=[
            {"role": "system", "content": sys_prompt_content},
            {"role": "user", "content": input},
            ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
    )
    
    response = response.dict()
    
    import time
    time.sleep(8)
    
    if n == 1:
        return get_output(response["choices"][0])
    return [get_output(res) for res in response['choices']]

def decoder_for_local_chat(model_name, input, max_tokens, n=1, stop=None, **kwargs):
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.95)
    frequency_penalty = kwargs.get('frequency_penalty', 0)
    presence_penalty = kwargs.get('presence_penalty', 0)

    unset_proxy()
    response = openai.ChatCompletion.create(
        api_key="EMPTY",
        api_base=api_base,
        model=model_name,
        messages=[
            {"role": "user", "content": input},
            ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
    )
    if n == 1:
        return get_output(response["choices"][0])
    return [get_output(res) for res in response['choices']]


def decoder_for_local_completion(model_name, input, max_tokens, n=1, stop=None, **kwargs):
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.95)
    frequency_penalty = kwargs.get('frequency_penalty', 0)
    presence_penalty = kwargs.get('presence_penalty', 0)

    unset_proxy()
    response = openai.Completion.create(
        api_key="EMPTY",
        api_base=api_base,
        model=model_name,
        prompt=input,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
    )
    if n == 1:
        return get_output_raw(response["choices"][0])
    return [get_output_raw(res) for res in response['choices']]
