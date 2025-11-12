import asyncio
import os

from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from IPython.display import display, Markdown
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

load_dotenv()

class BasketballAgent:
    def __init__(self, system_prompt: str, model: str = "gpt-4.1"):
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
    
    def get_message_count(self) -> int:
        """Get the number of messages in the conversation (excluding system)."""
        return len(self.__messages) - 1
    
    def clear_history(self, keep_system: bool = True):
        """Clear conversation history, optionally keeping the system prompt."""
        if keep_system:
            self.__messages = [self.__messages[0]]
        else:
            self.__messages = []

async def main():
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
    asyncio.run(main())