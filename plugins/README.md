# Ollama Streamlit Client Plugin Creation Guide

This guide will help you create plugins for Ollama Streamlit Client. Plugins allow you to extend the functionality of the Ollama Streamlit Client by adding custom processing steps or features.

## Plugin Structure

Each plugin should be a Python file with two main components:

1. A `plugin_info()` function that returns metadata about the plugin.
2. A `process()` function that contains the main logic of the plugin.

### plugin_info() Function

The `plugin_info()` function should return a dictionary with the following structure:

```python
def plugin_info():
    return {
        "description": "A brief description of what your plugin does",
        "params": {
            "param1": {
                "type": "text",
                "default": "default value",
                "help": "Description of param1"
            },
            "param2": {
                "type": "number",
                "default": 42,
                "help": "Description of param2"
            },
            "param3": {
                "type": "boolean",
                "default": False,
                "help": "Description of param3"
            }
        }
    }
```

The `params` dictionary defines the parameters that users can configure when using your plugin. Each parameter should have a type ("text", "number", or "boolean"), a default value, and a help text.

### process() Function

The `process()` function is where the main logic of your plugin goes. It should have the following signature:

```python
def process(model: str, messages: list, system_prompt: str, model_params: dict, generate_func: callable, **kwargs):
    # Your plugin logic here
    pass
```

Parameters:

- `model`: The current AI model being used
- `messages`: The conversation history
- `system_prompt`: The system prompt for the AI
- `model_params`: Parameters for the AI model
- `plugin_params`: The parameters configured by the user for this plugin
- `generate_func`: A function to generate AI responses
- `st`: The Streamlit object for UI interactions
- `ollama`: The Ollama object for interacting with the AI

## Example Plugin

Here's an example of a complete plugin that adds the current time to the system prompt:

```python
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

def process(model: str, messages: list, system_prompt: str, model_params: dict, plugin_params: dict, generate_func: callable, **kwargs):
    # Get the current time using the specified format
    time_format = plugin_params.get("time_format", "%Y-%m-%d %H:%M:%S %Z")
    current_time = datetime.datetime.now().strftime(time_format)

    # Add the current time to the system prompt
    updated_system_prompt = f"{system_prompt}\nCurrent date and time: {current_time}.".strip()

    # Generate response using the updated system prompt
    for chunk in generate_func(model, messages, updated_system_prompt, model_params):
        yield chunk
```

This plugin adds the current date and time to the system prompt, allowing the AI to be aware of the current time when generating responses. It also provides a configurable parameter for the time format.

## Best Practices

1. Keep your plugins focused on a single task or feature.
2. Provide clear and concise descriptions for your plugin and its parameters.
3. Handle errors gracefully and provide meaningful error messages.
4. Test your plugin thoroughly before sharing it.
5. Document any external dependencies your plugin might have.

## Submitting Your Plugin

[Include instructions on how users can submit or share their plugins with others, if applicable]

Happy plugin creation!
