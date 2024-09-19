import base64
import streamlit as st
import ollama
from typing import Dict, Generator
from streamlit import _bottom

APP_NAME = "Ollama Streamlit Client"


def ollama_generate_response(
    model_name: str, messages: Dict, system_prompt: str, params: Dict
) -> Generator:
    if system_prompt != "":
        messages = [{"role": "system", "content": system_prompt}] + messages
    print(
        f"[DEBUG] sending {len(messages)} messages to Ollama API, with params: {params}"
    )
    stream = ollama.chat(
        model=model_name, messages=messages, stream=True, options=params
    )
    for chunk in stream:
        if st.session_state.stop_stream:
            print("user stop stream")
            break
        yield chunk


def cb_reset_conversation():
    print("[DEBUG] reset conversation")
    init_session_state(force=True)


def cb_update_image_uploader_key():
    st.session_state["file_uploader_key"] += 1


def cb_stop_stream():
    print("[DEBUG] stop button callback")
    if st.session_state.last_response is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.last_response}
        )
        st.session_state.last_response = None
    st.session_state.stop_stream = True
    cb_enable_user_input()


def cb_disable_user_input():
    print("[DEBUG] disable user input")
    st.session_state.user_input_disabled = True


def cb_enable_user_input():
    print("[DEBUG] enable user input")
    st.session_state.user_input_disabled = False


def ui_display_metrics(metrics: dict | None, show=True):
    """Display token usage details."""
    print("[DEBUG] message metrics:", metrics)
    if metrics is None:
        return

    latency = round(metrics.get("eval_duration", 0) / 10**9, 1)
    tps = max(1, round(metrics.get("eval_count", 0) / latency))
    usage_info = [
        f"Input Tokens: {metrics.get('prompt_eval_count', 0)}",
        f"Output Tokens: {metrics.get('eval_count', 0)}",
        f"Latency: {latency} s",
        f"TPS: {tps}",
    ]
    if show:
        st.caption("ℹ️ " + " | ".join(usage_info))
    else:
        st.caption("ℹ️", help=" | ".join(usage_info))


@st.cache_data(ttl="1d")
def ollama_get_models():
    print("[DEBUG] ollama get models")
    models = sorted(ollama.list()["models"], key=lambda x: x["name"])
    # get model context length info
    for model in models:
        info = ollama.show(model["name"])
        family = info["details"]["family"]
        context_length = info["model_info"].get(f"{family}.context_length", None)
        # if context_length is None, convert it to k
        if context_length:
            context_length_k = int(context_length / 1024)
            model["context_length"] = context_length
            model["context_length_k"] = context_length_k
        model["size_gb"] = round(model["size"] / 1024**3, 1)
        # check if the model has a vision encoder
        model["has_vision_encoder"] = info.get("projector_info", {}).get(
            "clip.has_vision_encoder", False
        )
    return models


def ui_custom_css():
    print("[DEBUG] inject custom css")
    st.html(
        """
        <style>
            /** fix the button to the top of the sidebar **/
            .eczjsme9 {
                position: absolute; top: 0; padding: 0; width: 100%;
            }
            /** display user and assistant chat on opposite sides **/
            .stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
                flex-direction: row-reverse; text-align: right; justify-content: flex-end;
            }
            /** align image messages to the right **/
            .stImage.st-emotion-cache-1kyxreq.e115fcil2 {
                justify-content: flex-end;
            }
            .stImage.st-emotion-cache-1kyxreq.e115fcil2 img{
                max-height: 300px;
            }
            /** show vision icon if the model has a vision encoder **/
            .element-container:has(.hasvision) {
                width: 0; position: absolute; left: -55px; top: -6px;
            }
            /** show stop button above chat_input widget **/
            .e1f1d6gn4:nth-child(2):has(> .stButton) {
                position: absolute; text-align: right;
            }
            .e1f1d6gn4:nth-child(2):has(> .stButton) button {
                border: none; background: gainsboro;
            }
        </style>
        """
    )


def ui_sidebar(models, app_name):
    print("[DEBUG] show sidebar")
    with st.sidebar:
        st.html("<a href='/' target='_self'>🏠</a>")
        st.title(app_name)
        ui_model_selector(models)
        ui_system_prompt()
        ui_model_params()
        ui_new_conversation_button()


def ui_model_selector(models):
    options = [
        f"{model['name']} [{model['context_length_k']}k] [{model['size_gb']}GB]{' [👁️‍🗨️]' if model['has_vision_encoder'] else ''}"
        for model in models
    ]
    selected_model = ""
    selected_model_from_url = st.query_params.get("selected_model", "").strip()
    if st.session_state.get("select_model", ""):
        selected_model = st.session_state.select_model.split(" ")[0]
    else:
        selected_model = selected_model_from_url
    # fmt: off
    selected_model_idx = next((i for i, option in enumerate(options) if option.split(" ")[0] == selected_model), None)
    if selected_model_idx is not None:
        st.session_state.selected_model_info = models[selected_model_idx]
    selection = st.selectbox(
        "Model:",
        options,
        on_change=cb_reset_conversation,
        index=selected_model_idx,
        placeholder="Select a model",
        key="select_model",
    )
    if selection:
        new_selected_model = (
            selection.split(" ")[0] if selection is not None else ""
        )
        st.session_state.selected_model = new_selected_model
        if new_selected_model != selected_model_from_url:
            print("[DEBUG] update st.query_params.selected_model", new_selected_model)
            st.query_params["selected_model"] = new_selected_model


def ui_system_prompt():
    system_prompt = st.query_params.get("system_prompt", "")
    if "system_prompt" in st.session_state:
        system_prompt = st.session_state.system_prompt
    system_prompt = st.text_area(
        "System Prompt", key="system_prompt", value=system_prompt
    )
    # update system prompt in url query params
    if st.query_params.get("system_prompt", "") != system_prompt:
        if system_prompt == "":
            del st.query_params["system_prompt"]
        else:
            st.query_params["system_prompt"] = system_prompt


def ui_model_params():
    with st.expander("Model Parameters"):
        with st.container():
            temperature_help = "The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)"
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 2.0, 0.8, 0.1, help=temperature_help
            )

        with st.container():
            top_k_help = "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)"
            st.session_state.top_k = st.number_input(
                "Top K", 1, 100, 40, help=top_k_help
            )

        with st.container():
            top_p_help = "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)"
            st.session_state.top_p = st.slider(
                "Top P", 0.0, 1.0, 0.9, 0.1, help=top_p_help
            )

        with st.container():
            # Get the default num_ctx from the selected model info
            selected_model_info = st.session_state.get("selected_model_info", {})
            default_num_ctx = selected_model_info.get("context_length", 4096)
            max_num_ctx = max(32768, default_num_ctx)  # Ensure max is at least 32768
            num_ctx_help = "Sets the size of the context window used to generate the next token. (Default: same as selected model's context length)"
            # fmt: off
            st.session_state.num_ctx = st.number_input(
                "Context Window Size", 0, max_num_ctx, default_num_ctx, 512, help=num_ctx_help
            )

        with st.container():
            # Add num_predict parameter
            num_predict_help = "Same as OpenAI API max_tokens, maximum number of tokens to predict when generating text. (Default: 2048, -1 = infinite generation, -2 = fill context)"
            st.session_state.num_predict = st.number_input(
                "Number of Tokens to Predict", -2, 32768, 2048, help=num_predict_help
            )


def ui_new_conversation_button():
    if st.button("New Conversation", key="new_conversation"):
        print("[DEBUG] new conversation")
        cb_reset_conversation()
        # if uploaded image, update uploader key to reset the image upload widget
        if st.session_state.get(
            f"image_upload_{st.session_state['file_uploader_key']}", None
        ):
            cb_update_image_uploader_key()


def ui_display_message(message: dict):
    st.markdown(message["content"])
    if "images" in message and message["images"]:
        for image_base64 in message["images"]:
            image = base64.b64decode(image_base64)
            st.image(image)


def ui_show_image_uploader(container):
    # Add image upload widget
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 1
    uploaded_image = st.session_state.get(
        f"image_upload_{st.session_state['file_uploader_key']}", None
    )

    # if user has uploaded an image, and sent some user input, then update image uploader key in order show a new widget
    if "user_input" in st.session_state:
        if uploaded_image and st.session_state.get("user_input", None):
            cb_update_image_uploader_key()
    with container:
        st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            key=f"image_upload_{st.session_state['file_uploader_key']}",
            accept_multiple_files=False,
        )
    return uploaded_image


def ui_conversation_container():
    print("[DEBUG] show conversation history")
    conversation_container = st.container()
    with conversation_container:
        ui_display_system_prompt()
        ui_display_chat_history()
        st.empty()  # prevent showing a stale container
    return conversation_container


def ui_display_system_prompt():
    if st.session_state.system_prompt != "":
        with st.chat_message("system", avatar=":material/settings_account_box:"):
            st.markdown(st.session_state.system_prompt)


def ui_display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                ui_display_assistant_avatar()
            ui_display_message(message)
            if message["role"] == "assistant":
                ui_display_metrics(message.get("metrics"))


def ui_chat_input_disabled():
    disabled = True if st.session_state.get("selected_model", "") == "" else False
    if not disabled:
        if "user_input_disabled" in st.session_state:
            disabled = st.session_state.user_input_disabled
    return disabled


def ui_chat_input_area():
    chat_input_col, image_uploader_col = st.columns([9, 1])
    with chat_input_col:
        print("[DEBUG] show user input")
        prompt = st.chat_input(
            "How could I help you?",
            disabled=ui_chat_input_disabled(),
            key="user_input",
            on_submit=cb_disable_user_input,
        )
        if prompt:
            print("[DEBUG] show stop button")
            st.button(
                ":material/stop_circle:", key="stop_responding", on_click=cb_stop_stream
            )
    with image_uploader_col:
        with st.popover(
            ":material/add_photo_alternate:",
            disabled=(not model_has_vision() or ui_chat_input_disabled()),
        ):
            print("[DEBUG] show image uploader")
            image_uploader_container = st.container()
            uploaded_image = ui_show_image_uploader(image_uploader_container)

    return prompt, uploaded_image


def process_user_input(prompt, uploaded_image, conversation_container):
    if prompt:
        print("[DEBUG] process user input:", prompt)
        st.session_state.stop_stream = False
        message = create_user_message(prompt, uploaded_image)
        st.session_state.messages.append(message)

        with conversation_container:
            ui_display_user_message(message)
            ui_display_assistant_response(message)

        if st.session_state.user_input_disabled:
            cb_enable_user_input()
            # Rerun so that the chat_input() will be rendered enabled
            print("[DEBUG] st.rerun()")
            st.rerun()


def create_user_message(prompt, uploaded_image):
    message = {"role": "user", "content": prompt}
    if uploaded_image:
        print("[DEBUG] user uploaded image: ", uploaded_image)
        image_bytes = uploaded_image.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message["images"] = [image_base64]
    return message


def ui_display_user_message(message):
    with st.chat_message("user"):
        ui_display_message(message)


def ui_display_assistant_avatar():
    if model_has_vision():
        st.html('<div class="hasvision">👁️‍🗨️</div>')


def ui_display_assistant_response(message):
    with st.chat_message("assistant"):
        ui_display_assistant_avatar()
        msg_holder = st.empty()
        full_response, metrics = generate_assistant_response(msg_holder)
        msg_holder.markdown(full_response)
        ui_display_metrics(metrics)

    if full_response != "":
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "metrics": metrics}
        )


def generate_assistant_response(msg_holder):
    full_response = ""
    metrics = None
    params = {
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "top_k": st.session_state.top_k,
        "num_ctx": st.session_state.num_ctx,
        "num_predict": st.session_state.num_predict,
    }
    with st.spinner(""):
        for trunk in ollama_generate_response(
            st.session_state.selected_model,
            st.session_state.messages,
            st.session_state.system_prompt,
            params,
        ):
            if trunk["done"]:
                keys = [
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                ]
                metrics = {k: trunk.get(k, 0) for k in keys}
            else:
                response = trunk["message"]["content"]
            full_response += response
            st.session_state.last_response = full_response
            msg_holder.markdown(full_response + "▌")
    return full_response, metrics


def init_session_state(force=False):
    print("[DEBUG] init session state")
    if force or "messages" not in st.session_state:
        st.session_state.messages = []
    if force or "stop_stream" not in st.session_state:
        st.session_state.stop_stream = False
    if force or "last_response" not in st.session_state:
        st.session_state.last_response = None
    if force or "user_input_disabled" not in st.session_state:
        st.session_state.user_input_disabled = False


def model_has_vision():
    model_info = st.session_state.get("selected_model_info", None)
    return model_info and model_info["has_vision_encoder"]


def main():
    print("[DEBUG] main()")
    ui_custom_css()
    init_session_state()

    models = ollama_get_models()
    ui_sidebar(models, APP_NAME)

    conversation_container = ui_conversation_container()

    with _bottom:
        prompt, uploaded_image = ui_chat_input_area()
        process_user_input(prompt, uploaded_image, conversation_container)


if __name__ == "__main__":
    main()
