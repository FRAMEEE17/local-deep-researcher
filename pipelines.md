## Pipelines: UI-Agnostic OpenAI API Plugin Framework
warning
- DO NOT USE PIPELINES IF!
If your goal is simply to add support for additional providers like Anthropic or basic filters, you likely don't need Pipelines . For those cases, Open WebUI Functions are a better fit—it's built-in, much more convenient, and easier to configure. Pipelines, however, comes into play when you're dealing with computationally heavy tasks (e.g., running large models or complex logic) that you want to offload from your main Open WebUI instance for better performance and scalability.

Welcome to Pipelines, an Open WebUI initiative. Pipelines bring modular, customizable workflows to any UI client supporting OpenAI API specs – and much more! Easily extend functionalities, integrate unique logic, and create dynamic workflows with just a few lines of code.

🚀 Why Choose Pipelines?
Limitless Possibilities: Easily add custom logic and integrate Python libraries, from AI agents to home automation APIs.
Seamless Integration: Compatible with any UI/client supporting OpenAI API specs. (Only pipe-type pipelines are supported; filter types require clients with Pipelines support.)
Custom Hooks: Build and integrate custom pipelines.
Examples of What You Can Achieve:
Function Calling Pipeline: Easily handle function calls and enhance your applications with custom logic.
Custom RAG Pipeline: Implement sophisticated Retrieval-Augmented Generation pipelines tailored to your needs.
Message Monitoring Using Langfuse: Monitor and analyze message interactions in real-time using Langfuse.
Rate Limit Filter: Control the flow of requests to prevent exceeding rate limits.
Real-Time Translation Filter with LibreTranslate: Seamlessly integrate real-time translations into your LLM interactions.
Toxic Message Filter: Implement filters to detect and handle toxic messages effectively.
And Much More!: The sky is the limit for what you can accomplish with Pipelines and Python. Check out our scaffolds to get a head start on your projects and see how you can streamline your development process!
🔧 How It Works
Pipelines Workflow

Integrating Pipelines with any OpenAI API-compatible UI client is simple. Launch your Pipelines instance and set the OpenAI URL on your client to the Pipelines URL. That's it! You're ready to leverage any Python library for your needs.

⚡ Quick Start with Docker
warning
Pipelines are a plugin system with arbitrary code execution — don't fetch random pipelines from sources you don't trust.

For a streamlined setup using Docker:

Run the Pipelines container:

docker run -d -p 9097:9097 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main


Connect to Open WebUI:

Navigate to the Admin Panel > Settings > Connections section in Open WebUI.
When you're on this page, you can press the + button to add another connection.
Set the API URL to http://localhost:9097 and the API key to 0p3n-w3bu!.
Once you've added your pipelines connection and verified it, you will see an icon appear within the API Base URL field for the added connection. When hovered over, the icon itself will be labeled Pipelines. Your pipelines should now be active.
info
If your Open WebUI is running in a Docker container, replace localhost with host.docker.internal in the API URL.

Manage Configurations:

In the admin panel, go to Admin Panel > Settings > Pipelines tab.
Select your desired pipeline and modify the valve values directly from the WebUI.
tip
If you are unable to connect, it is most likely a Docker networking issue. We encourage you to troubleshoot on your own and share your methods and solutions in the discussions forum.

If you need to install a custom pipeline with additional dependencies:

Run the following command:

docker run -d -p 9097:9097 --add-host=host.docker.internal:host-gateway -e PIPELINES_URLS="https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py" -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main


Alternatively, you can directly install pipelines from the admin settings by copying and pasting the pipeline URL, provided it doesn't have additional dependencies.

That's it! You're now ready to build customizable AI integrations effortlessly with Pipelines. Enjoy!

📦 Installation and Setup
Get started with Pipelines in a few easy steps:

Ensure Python 3.11 is installed. It is the only officially supported Python version.

Clone the Pipelines repository:
```
git clone https://github.com/open-webui/pipelines.git
cd pipelines
```

Install the required dependencies:
```
pip install -r requirements.txt
```
Start the Pipelines server:
```
sh ./start.sh
```

Once the server is running, set the OpenAI URL on your client to the Pipelines URL. This unlocks the full capabilities of Pipelines, integrating any Python library and creating custom workflows tailored to your needs.

📂 Directory Structure and Examples
The /pipelines directory is the core of your setup. Add new modules, customize existing ones, and manage your workflows here. All the pipelines in the /pipelines directory will be automatically loaded when the server launches.

You can change this directory from /pipelines to another location using the PIPELINES_DIR env variable.

Integration Examples
Find various integration examples in the https://github.com/open-webui/pipelines/blob/main/examples directory. These examples show how to integrate different functionalities, providing a foundation for building your own custom pipelines.