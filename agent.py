import asyncio
import os

from typing import Any, Dict, List, Generator
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import gradio as gr

from IPython.display import display, Markdown
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

load_dotenv()

class BasketballAgent:
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.__client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.__system_prompt = system_prompt
        self.__model = model
        self.__messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": self.__system_prompt}]

    def set_system_prompt(self, prompt: str) -> None:
        self.__system_prompt = prompt
        self.__messages[0] = {"role": "system", "content": self.__system_prompt}

    def get_system_prompt(self) -> str:
        return self.__system_prompt
    
    def set_model(self, model: str) -> None:
        self.__model = model

    def get_model(self) -> str:
        return self.__model
    
    def send_message(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        # Add user message to history
        self.__messages.append({"role": "user", "content": user_message})
        
        # Get response from OpenAI
        response = self.__client.chat.completions.create(
            model=self.__model,
            messages=self.__messages
        )
        
        # Extract assistant's response
        assistant_message = response.choices[0].message.content or ""
        
        # Add assistant's response to history
        self.__messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def send_message_stream(self, user_message: str) -> Generator[str, None, None]:
        """Send a message and stream the response, maintaining conversation history."""
        # Add user message to history
        self.__messages.append({"role": "user", "content": user_message})
        
        # Get streaming response from OpenAI
        stream = self.__client.chat.completions.create(
            model=self.__model,
            messages=self.__messages,
            stream=True
        )
        
        # Collect the full response while yielding chunks
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # Add assistant's complete response to history
        self.__messages.append({"role": "assistant", "content": full_response})
    
    def get_message_count(self) -> int:
        """Get the number of messages in the conversation (excluding system)."""
        return len(self.__messages) - 1
    
    def clear_history(self, keep_system: bool = True):
        """Clear conversation history, optionally keeping the system prompt."""
        if keep_system:
            self.__messages = [self.__messages[0]]
        else:
            self.__messages = []

def create_gradio_interface():
    """Create a Gradio ChatInterface for the Basketball Agent."""
    system_prompt = """You are an NBA analyst agent. You know more about the current NBA and it's history than anyone else.
                        You are able to answer any question about the NBA, its players, teams, and history."""
    
    # Create a fresh agent for each session
    # Note: In production, you'd want session management for multi-user support
    agent = BasketballAgent(system_prompt=system_prompt)
    
    def chat_function(message, history):
        """Gradio chat function that streams responses.
        
        Args:
            message: The current user message
            history: List of [user_msg, assistant_msg] pairs from Gradio UI
                    We rely on our agent's internal history instead
        """
        # Stream the response chunk by chunk
        response = ""
        for chunk in agent.send_message_stream(message):
            response += chunk
            yield response
    
    # Create the ChatInterface
    interface = gr.ChatInterface(
        fn=chat_function,
        title="🏀 Basketball Agent",
        description="Ask me anything about the NBA - players, teams, stats, and history!",
        examples=[
            "Who is the all-time leading scorer in NBA history?",
            "Tell me about the 1996 Bulls championship run",
            "Who won MVP in 2023?",
            "Compare LeBron James and Michael Jordan"
        ],
    )
    
    return interface

async def main_cli():
    """Command-line interface version."""
    system_prompt = """You are an NBA analyst agent. You know more about the current NBA and it's history than anyone else.
                        You are able to answer any question about the NBA, its players, teams, and history. Respond in Markdown format."""
    
    chat = BasketballAgent(system_prompt=system_prompt)
    console = Console()
    
    print("Hello, I am your personal basketball agent. I know all about the NBA. Ask me anything (type 'quit' to exit):\n")
    
    while True:
        user_prompt = input("You: ")
        print()

        match user_prompt.lower():
            case "help":
                help_text = """
                **Available Commands:**
                - `help`: Show this help message.
                - `message count`: Show the number of messages in the conversation.
                - `clear history`: Clear the conversation history.
                - `show system prompt`: Display the current system prompt.
                - `show model`: Display the current model being used.
                - `change model`: Change the model being used.
                - `quit`: Exit the chat.
                """
                console.print(RichMarkdown(help_text))
                print()
                continue
            case "clear history":
                chat.clear_history()
                console.print(RichMarkdown("**Conversation history cleared.**"))
                print()
                continue
            case "show system prompt":
                console.print(RichMarkdown(f"**System Prompt:** {chat.get_system_prompt()}"))
                print()
                continue
            case "show model":
                console.print(RichMarkdown(f"**Current Model:** {chat.get_model()}"))
                print()
                continue
            case "change model":
                new_model = input("Enter new model name: ")
                chat.set_model(new_model)
                console.print(RichMarkdown(f"**Model changed to:** {chat.get_model()}"))
                print()
                continue
            case "quit":
                console.print(RichMarkdown("**Goodbye!**"))
                print()
                break

        response = chat.send_message(user_prompt)
        console.print(RichMarkdown(response))
        print()  # Add blank line for readability

if __name__ == "__main__":
    import sys
    
    # Check if user wants Gradio interface or CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        asyncio.run(main_cli())
    else:
        # Launch Gradio interface by default
        interface = create_gradio_interface()
        interface.launch()