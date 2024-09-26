def plugin_info():
    return {
        "description": "Zero-shot-CoT prompting(paper https://arxiv.org/abs/2205.11916), simply add the 'Let's think step by step.' to the end of the user's query",
    }


# fmt: off
def process(model: str, messages: list, system_prompt: str, model_params: dict, generate_func: callable, **kwargs):
    cot_prompt = "Let's think step by step."
    messages = messages + [{"role": "user", "content": cot_prompt}]

    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk
