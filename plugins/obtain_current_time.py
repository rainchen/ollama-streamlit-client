import datetime


def plugin_info():
    return {
        "description": "Adds the current time to the system prompt",
        "params": {
            "time_format": {
                "type": "text",
                "default": "%Y-%m-%d %H:%M:%S %Z",
                "help": "Format string for the current time (e.g., '%Y-%m-%d %H:%M:%S %Z' for '2023-04-15 14:30:00 UTC')",
            }
        },
    }


# fmt: off
def process(model: str, messages: list, system_prompt: str, model_params: dict, plugin_params: dict, generate_func: callable, **kwargs):
    # Get the current time using the specified format
    time_format = plugin_params.get("time_format", "%Y-%m-%d %H:%M:%S %Z")
    current_time = datetime.datetime.now().strftime(time_format)
    
    # Add the current time to the system prompt
    updated_system_prompt = f"{system_prompt}\nCurrent date and time: {current_time}.".strip()
    
    # Generate response using the updated system prompt
    for chunk in generate_func(model, messages, updated_system_prompt, model_params):
        yield chunk
