# uses a Chain of Thought (CoT) approach with reflection to answer queries
# refs: https://github.com/codelion/optillm/blob/main/optillm/cot_reflection.py
from . import cot_reflection_prompting

PLUGIN_MESSAGE_ID = "plugin.cot_reflection_reasoning.reasoning_result"


def plugin_info():
    return {
        "description": "Uses a Chain of Thought (CoT) approach with reflection to answer queries",
        "params": {
            "display_reasoning": {
                "type": "boolean",
                "default": False,
                "help": "Determines if the reasoning process should be shown",
            },
        },
    }


# fmt: off
def process(model: str, messages: list, system_prompt: str, model_params: dict, plugin_params: dict, generate_func: callable, st, **kwargs):
    initial_query = messages[-1]["content"]
    # skip PLUGIN_MESSAGE_ID messages
    filtered_messages = [msg for msg in messages if msg.get("name") != PLUGIN_MESSAGE_ID]
    cot_reflection_generator = cot_reflection_prompting.process(model, filtered_messages, system_prompt, model_params, generate_func)
    display_reasoning = plugin_params.get("display_reasoning", False)
    if display_reasoning:
        reasoning, metrics = process_generator(cot_reflection_generator, st)
        message = {"role": "assistant", "content": reasoning, "metrics": metrics, "name": PLUGIN_MESSAGE_ID}
        messages.append(message)
    else:
        reasoning, metrics = process_generator(cot_reflection_generator, None)

    reasoning_result = f"Following is my reasoning result:\n{reasoning}\n\nBased on my reasoning result, I provide my answer:\n".strip()
    
    messages=[
        {"role": "user", "content": initial_query},
        {"role": "assistant", "content": reasoning_result},
    ]
    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk


def process_generator(generator, st):
    full_response = ""
    metrics = {}
    msg_holder = st.empty() if st else None
    for chunk in generator:
        if chunk.get("done", False):
            metrics = {
                k: chunk.get(k, 0)
                for k in ["total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]
            }
        else:
            response = chunk["message"]["content"]
            full_response += response
            if msg_holder:
                msg_holder.markdown(full_response)
    return full_response, metrics
