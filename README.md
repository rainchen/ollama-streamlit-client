## Ollama Streamlit Client

### Introduction

Welcome to the **Ollama Streamlit Client**! This application is designed to provide a seamless and interactive interface for interacting with Ollama models directly through a simple Streamlit app. Whether you're a developer looking to integrate Ollama models into your projects or a user interested in exploring the capabilities of these models, this client offers a user-friendly experience with a range of features to enhance your interaction.

### Key Features

- **Auto List Available Ollama Models**: The client automatically lists all available Ollama models, making it easy to select and interact with the model that best suits your needs.
- **Bookmarkable URL for Selected Model**: The client generates a bookmarkable URL for the selected model, allowing you to easily share or revisit the specific model configuration.
- **System Prompt Customization**: Auto fetch the default system prompt from Ollama model info, and you can set a system prompt for each conversation, allowing you to customize the context and behavior of the model during your interactions.
- **Support Setting Model Parameters**: The client supports setting model parameters such as temperature, top_p, top_k, num_ctx, and num_predict, giving you more control over the model's behavior and output.
- **Quick Reset Conversation**: Quickly reset the conversation with a one-click button, allowing you to start a new interaction without reloading the application.
- **Message Metrics**: The client shows detailed message metrics, including Input Tokens, Output Tokens, Latency, and TPS (Tokens Per Second), providing insights into the performance and efficiency of the model.
- **Image Upload Capability**: For models that have vision capabilities, the client allows you to upload image files, enabling richer and more diverse interactions.
- **Responsive Interface**: The chat input is disabled while the model is responding, preventing multiple submissions and ensuring a smooth user experience.
- **Interrupt Responding**: A stop button is provided above the chat input widget, allowing you to interrupt the model's response if needed.

### Getting Started

To get started with the Ollama Streamlit Client, follow these steps:

#### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/rainchen/ollama-streamlit-client.git
   cd ollama-streamlit-client
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run ollama-streamlit-client.py
   ```

Once set up, you can launch the application and start exploring the capabilities of Ollama models through an intuitive and interactive interface.

### Usage Demo

<center class="half">
<img width="500" alt="demo-1-fs8" src="https://github.com/user-attachments/assets/5db9ec60-e308-4934-b6a5-2a0d63cabf8c"> <img width="500" alt="demo-2-fs8" src="https://github.com/user-attachments/assets/0d4d28e1-10d5-4a0d-9c73-2a143f3f0581">
</center>

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
