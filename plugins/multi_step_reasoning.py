# multi-step reasoning plugin for solving problems


def plugin_info():
    return {
        "description": "multi-step reasoning plugin for solving problems",
        "params": {
            "display_reasoning_process": {
                "type": "boolean",
                "default": True,
                "help": "Determines if the reasoning process should be shown",
            },
        },
    }


# fmt: off
MAX_ATTEMPTS = 3
def process(model: str, messages: list, system_prompt: str, model_params: dict, plugin_params: dict, generate_func: callable, st, **kwargs):
    problem = messages[-1]["content"]
    messages = solve_problem(model, problem, max_attempts=MAX_ATTEMPTS, generate_func=generate_func)

    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk

    if plugin_params.get("display_reasoning_process", False):
        st.session_state.messages.extend(messages)

def get_full_response(model: str, messages: list, system_prompt: str, model_params: dict, generate_func: callable):
    generator = generate_func(
                model,
                messages,
                system_prompt,
                model_params,
            )
    full_response = ""
    metrics = {}
    for chunk in generator:
        if chunk.get("done", False):
            # fmt: off
            metrics = {
                k: chunk.get(k, 0)
                for k in ["total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]
            }
        else:
            response = chunk["message"]["content"]
            full_response += response
    print("full_response: ", repr(full_response))
    print("metrics: ", metrics)
    return full_response, metrics

def solve_problem(model, problem: str, max_attempts: int = 10, generate_func: callable = None):
    """
    Solve a problem using multi-step reasoning, planning, and intelligent thinking.
    """
    reasoning_process = {
        "initial_problem": problem,
        "steps": [],
        "final_answer": "",
    }
    attempts = 0
    is_completed = False

    # Step 1: Analyze the problem and plan
    analysis_prompt = f"""
You are an AI assistant that excels at solving complex STEM problems using multi-step reasoning.
When given a problem, first analyze it, think about possible solution methods, and plan the subsequent steps to solve it.

Problem:
{problem}

Provide your analysis and step-by-step plan in plain text.
"""

    print("analysis_prompt: " + repr(analysis_prompt))
    messages = [{"role": "user", "content": analysis_prompt}]
    response, _ = get_full_response(model, messages, "", {}, generate_func)

    # Display AI's initial analysis
    print(f"### AI Initial Analysis:\n{response}\n")

    hint = response
    analysis_step = {"step_answer": "", "is_completed": False, "hint": hint}
    reasoning_process["steps"].append(analysis_step)

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant continuing the problem-solving process.",
        },
        {"role": "user", "content": "Giving a thought about this problem: " + problem},
        {"role": "assistant", "content": hint},
        {
            "role": "user",
            "content": "Solve it with this thought, and give the final answer",
        },
    ]

    # Continue with the plan and attempt to solve the problem
    while not is_completed and attempts < max_attempts:
        attempts += 1

        # Phase 1: Generate the step answer based on the thought
        response, _ = get_full_response(model, messages, "", {}, generate_func)

        # Extract step answer
        step_answer = response.strip()
        print(f"### Step Answer (Attempt {attempts}):\n{step_answer}\n")

        # Phase 2: Validate the step answer using XML format
        validation_prompt = f"""
You are an AI validator. Check if the following step answer solves the problem correctly:

Problem:
{problem}

Step Answer:
{step_answer}

Respond in XML format as follows:
<response>
    <is_correct>Is this answer 100% correct? Return true or false</is_correct>
    <hint>If the answer is incorrect, provide a new thought or hint.</hint>
</response>
"""

        print(f"**AI is validating step answer (Attempt {attempts})...**")
        messages_validation = [{"role": "user", "content": validation_prompt}]
        response, _ = get_full_response(model, messages_validation, "", {}, generate_func)
        print("AI response: ", repr(response))

        # Parse the XML response
        try:
            is_correct = "true" in response.lower()
            hint_start = response.find("<hint>") + len("<hint>")
            hint_end = response.find("</hint>")
            hint = (
                response[hint_start:hint_end].strip()
                if hint_start != -1 and hint_end != -1
                else "No hint provided"
            )
        except:
            is_correct = False
            hint = "Error parsing validation response."

        # Update reasoning process
        step = {"step_answer": step_answer, "is_completed": is_correct, "hint": hint}
        reasoning_process["steps"].append(step)

        messages += [{"role": "assistant", "content": step_answer}]

        if is_correct:
            break  # Exit loop if the step answer is correct

        messages += [{"role": "user", "content": "Not correct, try with this: " + hint}]

    # Final answer step
    messages += [
        {
            "role": "user",
            "content": f"Based on your reasoning, provide the final answer to the problem and return it in the same language as the following: {reasoning_process['initial_problem']}",
        }
    ]
    return messages
