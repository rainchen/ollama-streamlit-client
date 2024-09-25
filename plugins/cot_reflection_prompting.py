# uses a Chain of Thought (CoT) approach with reflection to answer queries
# refs: https://github.com/codelion/optillm/blob/main/optillm/cot_reflection.py
# fmt: off
import textwrap

def process(model: str, messages: list, system_prompt: str, model_params: dict, generate_func: callable, **kwargs):
    cot_prompt = textwrap.dedent(f"""
        {system_prompt}

        You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:

        1. Think through the problem step by step within the <thinking> tags.
        2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
        3. Make any necessary adjustments based on your reflection.
        4. Provide your final, concise answer within the <output> tags.

        Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
        Do not include any part of the final answer in these sections. 
        The actual response to the query must be entirely contained within the <output> tags.

        Use the following format for your response:
        <thinking>
        [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
        <reflection>
        [Your reflection on your reasoning, checking for errors or improvements]
        </reflection>
        [Any adjustments to your thinking based on your reflection]
        </thinking>
        <output>
        [Your final, concise answer to the query. This is the only part that will be shown to the user.]
        </output>
    """).strip()
        

    # if first messages is system prompt
    if messages[0].get("role") == "system":
         # add cot_prompt into system prompt once only
        if cot_prompt not in system_prompt:
            system_prompt = cot_prompt
    else:
        system_prompt = cot_prompt
    
    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk
