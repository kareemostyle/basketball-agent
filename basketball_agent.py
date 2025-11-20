import os
from typing import List, Generator
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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
