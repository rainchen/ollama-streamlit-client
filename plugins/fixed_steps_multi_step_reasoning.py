import json
import streamlit as st
import time
import ollama

PLUGIN_ID = __file__.split("/")[-1].split(".")[0]
PLUGIN_MESSAGE_ID = f"plugin.{PLUGIN_ID}.reasoning_result"

DEFAULT_REASONING_PROMPT = """
You are an AI language model engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.

Problem-Solving Steps:

1. Understand: Read and comprehend the user's question.
2. Basics: Identify fundamental concepts involved.
3. Break Down: Divide the problem into smaller parts.
4. Analyze: Use facts and data to examine each part.
5. Build: Assemble insights into a coherent solution.
6. Edge Cases: Consider and address exceptions.
7. Communicate: Present the solution clearly.
8. Verify: Review and reflect on the solution.
""".strip()


def plugin_info():
    return {
        "description": "Follows fixed step instructions, generates step-by-step reasoning chains to solve user problems through first-principles thinking and evidence-based reasoning.",
        "params": {
            "display_reasoning": {
                "type": "boolean",
                "default": True,
                "help": "Whether to display the reasoning process.",
            },
            "reasoning_prompt": {
                "type": "text",
                "default": DEFAULT_REASONING_PROMPT,
                "height": 160,
                "help": "The system prompt to use for the reasoning process.",
            },
            "max_steps": {
                "type": "number",
                "min_value": 1,
                "max_value": 30,
                "step": 1,
                "default": 10,
                "help": "Maximum number of reasoning steps allowed.",
            },
        },
    }


# fmt: off
def process(model: str, messages: list, system_prompt: str, model_params: dict, plugin_params: dict, generate_func: callable, **kwargs):
    display_reasoning = plugin_params.get("display_reasoning", False)
    reasoning_prompt = plugin_params.get("reasoning_prompt", DEFAULT_REASONING_PROMPT)
    max_steps = plugin_params.get("max_steps", 10)

    # Create empty elements to hold the generated text and total time
    response_container = st.empty()
    time_container = st.empty()
    
    user_query = messages[-1]["content"]
    reasoning_process = []
    total_reasoning_time = None
    # Generate and display the response
    for steps, total_thinking_time in generate_response(user_query, reasoning_prompt, model, max_steps):
        with response_container.container():
            for i, (title, content, thinking_time, metrics) in enumerate(steps):
                if display_reasoning:
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(str(content).replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            reasoning_process.append((f"**{title}**\n\n{content}\n\n", metrics))
        
        # Only show total time when it's available at the end
        if total_thinking_time is not None:
            total_time_info = f"**Total thinking time: {total_thinking_time:.2f} seconds**"
            if display_reasoning:
                time_container.markdown(total_time_info)
            total_reasoning_time = total_thinking_time

    if display_reasoning:
        for i, (content, metrics) in enumerate(reasoning_process):
            # add total time info to the last step
            if i == len(reasoning_process) - 1 and total_reasoning_time is not None:
                content += f"\n\n{total_time_info}"
            message = {"role": "assistant","content": content,"name": PLUGIN_MESSAGE_ID, "metrics": metrics}
            messages.append(message)
    reasoning_result = ''.join([content for content, _ in reasoning_process])
    reasoning_result_prompt = f"Following is my reasoning:\n{reasoning_result}\n\nBased on my reasoning, I provide my answer:\n"
    messages = [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": reasoning_result_prompt, "metrics": metrics},
    ]
    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk


def make_api_call(model, messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            metrics = None
            if attempt != 0:
                print_debug(f"retry making api call {attempt}")
            print_debug("send messages:\n", messages)
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature":0.2, "max_length":max_tokens},
                format='json',
            )
            print_debug("get response:\n", response)
            metrics = {
                k: response.get(k, 0)
                for k in ["total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]
            }
            # ensure step data structure is valid
            step_data = json.loads(response["message"]["content"])
            valid_keys = ['title', 'content', 'next_action']
            if is_final_answer:
                valid_keys.remove('next_action')
            if not all(key in step_data for key in valid_keys):
                raise ValueError("Invalid step data")
            return step_data, metrics
        except Exception as e:
            print_debug(f"error making api call: {str(e)}")
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}, metrics
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}, metrics
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(user_query, reasoning_prompt, model, max_steps):
    messages = [
        {
            "role": "system",
            "content": """
{reasoning_prompt}

Make sure all steps are included.
For each step, provide a title and content. Respond in JSON format with 'title', 'content', and 'next_action' (must be either 'continue' or 'final_answer') keys.
Example of a valid JSON response:
```json
{
    "title": "the first step title: purpose of the first step",
    "content": "reasoning of the first step",
    "next_action": "continue"
}```
Ensure all JSON keys are present and lowercase.
For example, when the last step is "Verify: Review and reflect on the solution.", the last step JSON response should be:
```json
{
    "title": "Verify: Review and reflect on the solution.",
    "content": "reasoning of the last step",
    "next_action": "final_answer"
}```

""".replace("{reasoning_prompt}", reasoning_prompt).strip(),
        },
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "content": "OK. I will now think step by step following my instructions, starting at the beginning after decomposing the problem.",
        },
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        print_debug(f"start step {step_count}, max {max_steps}")
        start_time = time.time()
        step_data, metrics = make_api_call(model, messages, 300)
        print_debug("get step_data:", repr(step_data))
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        if not step_data == {}:
            print_debug(f"append step {step_count}: {step_data['title']}")
            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time, metrics))
            messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        next_action = step_data.get("next_action", "").replace(" ", "_").lower() # in case the next_action is "Final Answer"
        if next_action == 'final_answer' or step_count >= max_steps:
            yield steps, None  # last step
            break
        else:
            # messages.append({"role": "user", "content": f"continue step {step_count+1}, ensure all defined steps are included."})
            messages.append({"role": "user", "content": f"continue step {step_count+1}, ensure all steps are outputed."})
        
        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    print_debug("get final answer")
    start_time = time.time()
    final_data, metrics = make_api_call(model, messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data.get("content", ""), thinking_time, metrics))

    yield steps, total_thinking_time

def print_debug(message, *args):
    print(f"\033[94m[DEBUG] plugins/{PLUGIN_ID}: {message}\033[0m", *args)
