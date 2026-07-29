import requests
import json
from typing import Optional, Dict, Any

class OllamaClient:
    """
    A simple client wrapper for interacting with the local Ollama API.
    It simplifies sending prompts to various models running locally.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initializes the connection details for the Ollama API."""
        self.base_url = base_url
        print(f"🔗 Initialized Ollama Client pointing to: {self.base_url}")

    def _request(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Internal method to handle all HTTP requests and error checking."""

        # Construct the full URL for the API call
        full_url = f"{self.base_url}/api/{endpoint}"

        try:
            response = requests.post(full_url, json=data)
            response.raise_for_status() # Raises an HTTPError for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.ConnectionError:
            print("-" * 60)
            print(f"🚨 ERROR: Could not connect to Ollama at {self.base_url}.")
            print("   Please ensure the Ollama service is running in the background.")
            print("-" * 60)
            return {"error": "Connection Error: Is Ollama running?"}
        except requests.exceptions.HTTPError as e:
            # Handles errors like 'model not found' or 'forbidden'
            try:
                detail = e.response.json()
                print(f"\n🚨 HTTP API ERROR ({e.response.status_code}):")
                print(f"   Detail: {detail.get('error', 'Unknown API error.')}")
            except json.JSONDecodeError:
                 print("   Raw Response Detail:", e.response.text)
            return {"error": f"HTTP Error: Check the model name or Ollama server status."}
        except requests.exceptions.RequestException as e:
            print(f"\n🚨 An unexpected error occurred during request: {e}")
            return {"error": str(e)}


    def generate_response(self, model: str, prompt: str) -> Optional[str]:
        """
        Sends a single prompt to the specified model and returns the generated text.

        Args:
            model (str): The name of the model (e.g., "llama2", "mistral").
            prompt (str): The question or prompt text to send.

        Returns:
            Optional[str]: The clean response text, or None if an error occurred.
        """
        print(f"\n💡 Sending prompt to '{model}'...")

        # We use the /generate endpoint for simple prompts
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False # Set to True if you want streaming output
        }

        result = self._request("generate", data=payload)

        if "error" in result:
            print(f"\n🛑 Operation failed. Details: {result['error']}")
            return None

        # The response structure is {'model': '...', 'created_at': ..., 'response': '...'}
        generated_text = result.get("response")
        if generated_text:
             print("\n✅ Successfully retrieved response.")
             return generated_text
        else:
            print("\n🛑 Error: No content received in the Ollama response payload.")
            return None


    def chat(self, model: str, messages: list[tuple[str, str]]) -> Optional[str]:
        """
        Simulates a chat interaction using the /api/chat endpoint (recommended for conversation).

        Args:
            model (str): The name of the model.
            messages (list[tuple[role, content]]): A list of message tuples
                                                 (e.g., ("user", "Hi",) , ("system", "You are a helpful bot",)).

        Returns:
            Optional[str]: The clean response text, or None if an error occurred.
        """
        print("\n🤖 Starting Chat Interaction...")

        # Ollama Chat API expects roles (user/assistant/system) and messages array
        chat_messages = [
            {"role": role.lower(), "content": content}
            for role, content in messages
        ]

        payload = {
            "model": model,
            "messages": chat_messages
        }

        result = self._request("chat", data=payload)

        if "error" in result:
            print(f"\n🛑 Operation failed. Details: {result['error']}")
            return None

        # The structure is similar to generate, but uses the message content
        generated_text = result.get("message", {}).get("content")
        if generated_text:
             print("\n✅ Successfully retrieved chat response.")
             return generated_text
        else:
            print("\n🛑 Error: No content received in the Ollama chat response payload.")
            return None


# --- USAGE EXAMPLE ---

if __name__ == "__main__":

    # 1. Initialize the client (it defaults to localhost:11434)
    ollama_client = OllamaClient()

    # NOTE: Make sure 'llama2' or 'mistral' are pulled using `ollama pull [model_name]` first!
    MODEL_NAME = "llama2"

    print("\n" + "="*70)
    print("--- EXAMPLE 1: Single Prompt Generation (Simple Question) ---")
    print("="*70)

    prompt_question = "Explain the concept of quantum entanglement to a fifth grader in three sentences."
    response_single = ollama_client.generate_response(MODEL_NAME, prompt_question)

    if response_single:
        print("\n\n==================== OPTIMA RESPONSE ====================")
        print(f"🚀 {response_single}")
        print("="*60)


    # --- Chat Example (Multi-turn conversation context) ---

    MODEL_NAME_CHAT = "mistral" # Use a model that supports chat well
    if MODEL_NAME_CHAT == "mistral":
         try:
            ollama_client.generate_response(MODEL_NAME_CHAT, "This is just to check connection.")
         except Exception as e:
             print(f"\nSkipping chat example because '{MODEL_NAME}' might not be available or configured properly. Error: {e}")

    # Only run the chat example if we successfully connected before
    if 'response_single' and response_single is not None:
        print("\n" + "="*70)
        print("--- EXAMPLE 2: Chat Context Conversation ---")
        print("="*70)

        # Define conversation history (Role, Content)
        chat_history = [
            ("system", "You are a friendly and helpful assistant that writes short poems."), # Sets the context/persona
            ("user", "Write me a four-line poem about Python programming."),              # The first user prompt
        ]

        response_chat = ollama_client.chat(MODEL_NAME, chat_history)

        if response_chat:
            print("\n\n==================== OPTIMA CHAT RESPONSE ====================")
            print(f"📜 {response_chat}")
            print("="*60)

### Key Features and How It Works

# 1.  **`OllamaClient` Class:** Encapsulates all the logic, making it clean to use. You just instantiate it (`client = OllamaClient()`) and call methods.
# 2.  **`_request()` Method (The Engine):** This private method handles the boilerplate networking code:
#     *   It uses `requests.post()` for API interaction.
#     *   Crucially, it includes robust `try...except` blocks to catch common errors like **"Connection Error"** (if Ollama is off) and **"HTTP API Error"** (if the model name is wrong).
# 3.  **`generate_response()`:** Optimized for single, stateless questions (`/api/generate`). This is best when you just need a quick answer without remembering previous turns in the conversation.
# 4.  **`chat()`:** Uses the dedicated `/api/chat` endpoint. This is **highly recommended** for any task that requires memory or multi-turn dialogue (e.g., "What was the capital of France?" followed by, "And what river runs through it?").
# 5.  **Role Structure:** By using `("system", ...)`, `("user", ...)` tuples in the chat function, you give the model context and define who is speaking, which dramatically improves the quality of conversational output.

### 🚀 Best Practices & Tips

# *   **Model Pulling:** If you use a model name (e.g., `"mistral"`) that isn't installed on your local Ollama instance, the client will fail with an HTTP error. **Always run `ollama pull [model_name]` in your terminal first.**
# *   **Streaming Output:** If you are building a GUI or want to display text as it is being generated (typing effect), change `stream: False` to `stream: True` in the `generate_response` payload, and you will need to modify the internal request handling to process streaming chunks.
# *   **System Prompts:** For complex tasks or defining a persona (e.g., "You are a pirate who only speaks in rhymes"), use the `chat()` method and place your instructions within the first message tuple: `("system", "Your response must be written as if you are Sir Reginald, a dramatic 18th-century nobleman.")
