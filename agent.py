import asyncio
import sys
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

from basketball_agent import BasketballAgent
from gui import create_gradio_interface

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
    # Check if user wants Gradio interface or CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        asyncio.run(main_cli())
    else:
        # Launch Gradio interface by default
        interface = create_gradio_interface()
        interface.launch()