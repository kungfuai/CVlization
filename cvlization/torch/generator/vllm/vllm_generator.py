"""
pip install vllm transformers==4.34.0 accelerate==0.20.3

vllm requires more recent GPU, otherwise it will throw an error like this:

ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your NVIDIA GeForce RTX 2080 Ti GPU has compute capability 7.5.
"""
from vllm import LLM
from vllm import SamplingParams


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>

{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT, instruction="{instruction}"
)


def gen_text(llm, prompts, use_template=False, **kwargs):
    llm = llm or LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    # the default max length is pretty small (16), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.8

    sampling_params = SamplingParams(**kwargs)

    outputs = llm.generate(full_prompts, sampling_params=sampling_params)
    texts = [out.outputs[0].text for out in outputs]

    return texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the meaning of life?")
    parser.add_argument("--llm", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--use_template", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    args = parser.parse_args()

    llm = LLM(model=args.llm, dtype="bfloat16")
    prompts = [args.prompt]
    texts = gen_text(
        llm,
        prompts,
        use_template=args.use_template,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
    )
    print(texts)
