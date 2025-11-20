import gradio as gr
from basketball_agent import BasketballAgent

def create_gradio_interface():
    """Create a Gradio ChatInterface for the Basketball Agent."""
    system_prompt = """You are an NBA analyst agent. You know more about the current NBA and it's history than anyone else.
                        You are able to answer any question about the NBA, its players, teams, and history."""
    
    # Create a fresh agent for each session
    # Note: In production, you'd want session management for multi-user support
    # We'll initialize the agent lazily or update it when the model changes
    agent_state = {"agent": BasketballAgent(system_prompt=system_prompt)}
    
    def chat_function(message, history):
        """Gradio chat function that streams responses.
        
        Args:
            message: The current user message
            history: List of [user_msg, assistant_msg] pairs from Gradio UI
                    We rely on our agent's internal history instead
        """
        # Stream the response chunk by chunk
        response = ""
        for chunk in agent_state["agent"].send_message_stream(message):
            response += chunk
            yield response

    def update_model(new_model):
        """Update the agent's model."""
        agent_state["agent"].set_model(new_model)
        return f"Model updated to {new_model}"

    # Define available models
    available_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview"
    ]

    with gr.Blocks() as interface:
        gr.Markdown("# 🏀 Basketball Agent")
        gr.Markdown("Ask me anything about the NBA - players, teams, stats, and history!")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value="gpt-4o-mini",
                label="Select Model",
                interactive=True
            )
            model_status = gr.Textbox(value="Model: gpt-4o-mini", label="Status", interactive=False)

        model_dropdown.change(fn=update_model, inputs=model_dropdown, outputs=model_status)

        chat_interface = gr.ChatInterface(
            fn=chat_function,
            examples=[
                "Who is the all-time leading scorer in NBA history?",
                "Tell me about the 1996 Bulls championship run",
                "Who won MVP in 2023?",
                "Compare LeBron James and Michael Jordan"
            ],
        )
    
    return interface
