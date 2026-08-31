"""
Ollama(gemma4) 호출 계층 — review2query.py[3]과 semantic_card.py[4]가 공유한다.

두 단계는 하는 일이 다르지만(리뷰 -> 쿼리 / 메타+리뷰 -> 카드) **Ollama를 쓰는 방식은 같아야
한다.** 예전에는 각자 구현이라 아래가 전부 달랐다:

  · 재시도     쿼리 3회(2s/4s 백오프) / 카드 0회
               -> 카드는 요청 하나가 실패하면 예외가 ex.map()에서 올라와 그 배치 32개가 통째로
                  죽었다. 카드가 더 긴 생성(300토큰)이라 타임아웃 위험이 더 큰데 보호는 더 약했다.
  · 동시성     쿼리 len(urls) * requests_per_server / 카드 len(urls) 고정(조절 불가)
  · URL 파싱   쿼리는 strip + 빈 값 제거 / 카드는 split(",")만
               -> "a, b"처럼 공백을 넣으면 카드 쪽만 " b"를 URL로 썼다
  · 진행 표시  쿼리는 tqdm 진행바 / 카드는 청크가 끝난 뒤 print 한 줄
  · 기본값     쿼리는 argparse에 문자열 하드코딩 / 카드는 config.CFG

그리고 semantic_card.py가 HTTP 헬퍼 하나를 얻으려고 파이프라인 이웃 단계인 review2query를
import하고 있었다.

여기 없는 것 = 단계마다 달라야 하는 것: 프롬프트 문안, max_new_tokens(쿼리 220 / 카드 300),
응답 후처리(clean_query / clean_card), 체크포인트 단위. 전부 호출자가 정한다.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

TIMEOUT_SEC = 120
MAX_RETRIES = 3          # 일시적 네트워크/모델 재로드 hiccup 한 번에 청크 전체가 죽지 않도록


def parse_urls(spec) -> list:
    """쉼표로 구분된 base URL 목록을 리스트로. 공백을 지우고 빈 항목은 버린다."""
    if isinstance(spec, (list, tuple)):
        items = spec
    else:
        items = str(spec).split(",")
    urls = [u.strip() for u in items if str(u).strip()]
    if not urls:
        raise ValueError(f"[ollama] 사용할 수 있는 base URL이 없습니다: {spec!r}")
    return urls


def generate(base_url: str, model: str, prompt: str, max_new_tokens: int,
             timeout: int = TIMEOUT_SEC) -> str:
    """단일 요청. 반환값은 후처리 전 원문(빈 문자열일 수 있다)."""
    r = requests.post(f"{base_url}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,   # gemma4는 reasoning 모델 — thinking에 예산을 다 쓰면 response가 빈 문자열로 잘림
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": max_new_tokens},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def generate_batch(base_urls, model: str, prompts: list, max_new_tokens: int,
                   desc: str, requests_per_server: int = 1) -> list:
    """프롬프트 목록을 서버들에 라운드로빈으로 분산 호출하고 원문 응답을 입력 순서대로 돌려준다.

    URL은 워커가 아니라 **프롬프트 인덱스**로 정하므로(base_urls[i % len]), 동시성을 바꿔도
    i번 프롬프트가 가는 서버는 달라지지 않는다. 후처리(clean_query/clean_card)는 순수 함수라
    호출자가 반환값에 적용한다 — 스레드 안에서 하던 것과 결과가 같다."""
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
