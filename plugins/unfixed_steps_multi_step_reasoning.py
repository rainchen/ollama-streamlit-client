import json
import streamlit as st
import time
import ollama

PLUGIN_MESSAGE_ID = "plugin.o1_like_multi_step_reasoning.reasoning_result"


def plugin_info():
    return {
        "description": "Generates O1-like reasoning chains for complex problem-solving",
        "params": {
            "display_reasoning": {
                "type": "boolean",
                "default": True,
                "help": "Whether to display the reasoning process.",
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
    max_steps = plugin_params.get("max_steps", 10)

    # Create empty elements to hold the generated text and total time
    response_container = st.empty()
    time_container = st.empty()
    
    user_query = messages[-1]["content"]
    reasoning_process = []
    total_reasoning_time = None
    # Generate and display the response
    for steps, total_thinking_time in generate_response(user_query, model, max_steps):
        with response_container.container():
            for i, (title, content, thinking_time, metrics) in enumerate(steps):
                if title.startswith("Final Answer"):
                    st.markdown(f"### {title}")
                    st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                else:
                    with st.expander(title, expanded=True):
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            reasoning_process.append((f"**{title}**\n\n{content}\n\n", metrics))
        
        # Only show total time when it's available at the end
        if total_thinking_time is not None:
            total_time_info = f"**Total thinking time: {total_thinking_time:.2f} seconds**"
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
            print("[DEBUG] plugins/unfixed_steps_multi_step_reasoning: sending messages to ollama:\n", messages)
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature":0.2, "max_length":max_tokens},
                format='json',
            )
            print("[DEBUG] plugins/unfixed_steps_multi_step_reasoning: response from ollama:\n", response)
            metrics = {
                k: response.get(k, 0)
                for k in ["total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]
            }
            return json.loads(response["message"]["content"]), metrics
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}, metrics
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}, metrics
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt, model, max_steps):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (must be either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
JSON keys must be present and lowercase.
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        print(f"[DEBUG] plugins/unfixed_steps_multi_step_reasoning: start step {step_count}, max {max_steps}")
        start_time = time.time()
        step_data, metrics = make_api_call(model, messages, 300)
        print("[DEBUG] plugins/unfixed_steps_multi_step_reasoning: step_data:", repr(step_data))
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        if not step_data == {}:
            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time, metrics))
            messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data.get("next_action", "") == 'final_answer' or step_count >= max_steps:
            yield steps, None # last step
            break
        else:
            messages.append({"role": "user", "content": "continue reasoning"})
        
        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data, metrics = make_api_call(model, messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data['content'], thinking_time, metrics))

    yield steps, total_thinking_time
