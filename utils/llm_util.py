import abc
import httpx
import logging
import numpy as np
from retry import retry
from pathlib import Path
from utils.io_util import load_json
from openai import OpenAI
from huggingface_hub import InferenceClient
import google.generativeai as genai


openai_keys_folder = Path(__file__).resolve().parent.parent / "openai_keys"
OPENAI_KEYS = load_json(openai_keys_folder / "openai_key.json")

gemini_keys_folder = Path(__file__).resolve().parent.parent / "gemini_keys"
GEMINI_KEYS = load_json(gemini_keys_folder / "gemini_keys.json")

logger = logging.getLogger(__name__)


@retry(tries=5, delay=60)
def connect_openai(
    client,
    engine,
    messages,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    # stop,
    response_format,
):
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
    )


class GPT_Chat:
    def __init__(
        self,
        engine,
        stop=None,
        max_tokens=1000,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        # add key
        self.client = OpenAI(
            api_key=OPENAI_KEYS["key"],
            organization=OPENAI_KEYS["org"],
            timeout=60,
            max_retries=5,
            http_client=httpx.Client(
                proxies=OPENAI_KEYS["proxy"],
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            ),
        )

    def get_response(
        self,
        prompt,
        messages=None,
        end_when_error=False,
        max_retry=2,
        temperature=0.0,
        force_json=False,
    ):
        conn_success, llm_output = False, ""
        if messages is not None:
            messages = messages
        else:
            messages = [{"role": "user", "content": prompt}]

        if force_json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        n_retry = 0
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                logger.info("[INFO] connecting to the LLM ...")

                response = connect_openai(
                    client=self.client,
                    engine=self.engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    response_format=response_format,
                )
                llm_output = response.choices[0].message.content
                conn_success = True
            except Exception as e:
                logger.info("[ERROR] LLM error: {}".format(e))
                if end_when_error:
                    break
        return conn_success, llm_output


class LLMBase(abc.ABC):
    def __init__(self, use_gpt_4: bool, *args, **kwargs):
        engine = "gpt-4-0125-preview" if use_gpt_4 else "gpt-3.5-turbo"
        self.llm_gpt = GPT_Chat(engine=engine)

    def prompt_llm(self, prompt: str, temperature: float = 0.0, force_json: bool = False):
        # feed prompt to llm
        logger.info("\n" + "#" * 50)
        logger.info(f"Prompt:\n{prompt}")
        messages = [{"role": "user", "content": prompt}]

        conn_success, llm_output = self.llm_gpt.get_response(
            prompt=None,
            messages=messages,
            end_when_error=False,
            temperature=temperature,
            force_json=force_json,
        )
        if not conn_success:
            raise Exception("Fail to connect to the LLM")

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output


class SLLMBase(abc.ABC):
    def __init__(self, use_gpt_4: bool, *args, **kwargs):
        # self.client = InferenceClient(
        #     "microsoft/Phi-3.5-mini-instruct",
        #     # "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     token='HUGGINGFACE_KEYS["key"]',
        # )

        genai.configure(api_key = GEMINI_KEYS["key"])
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    def prompt_llm(self, prompt: str, temperature: float = 0.0, force_json: bool = False):
        # feed prompt to llm
        logger.info("\n" + "#" * 50)
        logger.info(f"Prompt:\n{prompt}")

        # llm_output = ""
        # output = self.client.chat_completion(
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=512, ## Should be checked
        #     stream=True,
        # )

        # for message in output:
        #     print(message.choices[0].delta.content, end="")
        #     llm_output += message.choices[0].delta.content

        llm_output = self.model.generate_content(prompt).text
        
        start_index = llm_output.find('{\n')
        end_index = llm_output.find('\n}')

        # 텍스트가 존재하는지 확인하고 필터링
        if start_index != -1 and end_index != -1:
            llm_output = llm_output[start_index:end_index+2]

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output

####################### Convert Formats #####################


def textualize_array(array):
    return np.round(array, decimals=2).tolist()
