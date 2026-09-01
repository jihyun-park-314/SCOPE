import time
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

TIMEOUT_SEC = 120
MAX_RETRIES = 3

def parse_urls(spec) -> list:
    if isinstance(spec, (list, tuple)):
        items = spec
    else:
        items = str(spec).split(",")
    urls = [u.strip() for u in items if str(u).strip()]
    if not urls:
        raise ValueError(f"[ollama] no usable base URL: {spec!r}")
    return urls

def generate(base_url: str, model: str, prompt: str, max_new_tokens: int,
             timeout: int = TIMEOUT_SEC) -> str:
    r = requests.post(f"{base_url}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": max_new_tokens},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]

def generate_batch(base_urls, model: str, prompts: list, max_new_tokens: int,
                   desc: str, requests_per_server: int = 1) -> list:
    n = len(prompts)
    outs = [None] * n
    if n == 0:
        return outs
    base_urls = parse_urls(base_urls)
    pbar = tqdm(total=n, desc=desc, unit="req", dynamic_ncols=True)

    def work(i):
        url = base_urls[i % len(base_urls)]
        for attempt in range(MAX_RETRIES):
            try:
                outs[i] = generate(url, model, prompts[i], max_new_tokens)
                break
            except requests.exceptions.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        pbar.update(1)

    with ThreadPoolExecutor(max_workers=len(base_urls) * requests_per_server) as ex:
        list(ex.map(work, range(n)))
    pbar.close()
    return outs
