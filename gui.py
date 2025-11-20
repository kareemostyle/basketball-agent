import gradio as gr
from basketball_agent import BasketballAgent

def create_gradio_interface():
    """Create a Gradio ChatInterface for the Basketball Agent."""
    system_prompt = """You are an NBA analyst agent. You know more about the current NBA and it's history than anyone else.
                        You are able to answer any question about the NBA, its players, teams, and history."""
    
    def chat_function(message, history, agent_dict):
        """Gradio chat function that streams responses.
        
        Args:
            message: The current user message
            history: List of [user_msg, assistant_msg] pairs from Gradio UI
                    We rely on our agent's internal history instead
            agent_dict: Dict containing per-session agent instance (gr.State)
        """
        # Create agent on first use for this session
        if agent_dict is None:
            agent_dict = {}
        
        if "agent" not in agent_dict:
            agent_dict["agent"] = BasketballAgent(system_prompt=system_prompt, model="gpt-4o-mini")
        
        # Stream the response chunk by chunk
        response = ""
        for chunk in agent_dict["agent"].send_message_stream(message):
            response += chunk
            yield response

    def update_model(new_model, agent_dict):
        """Update the agent's model."""
        # Initialize dict if needed
        if agent_dict is None:
            agent_dict = {}
        
        # Create agent if it doesn't exist yet, otherwise update model
        if "agent" not in agent_dict:
            agent_dict["agent"] = BasketballAgent(system_prompt=system_prompt, model=new_model)
        else:
            agent_dict["agent"].set_model(new_model)
        
        return f"Model updated to {new_model}", agent_dict

    # Define available models
    available_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview"
    ]

    with gr.Blocks() as interface:
        # Create per-session state for the agent
        agent_state = gr.State(None)
        
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

        model_dropdown.change(fn=update_model, inputs=[model_dropdown, agent_state], outputs=[model_status, agent_state])

        chat_interface = gr.ChatInterface(
            fn=chat_function,
            additional_inputs=[agent_state],
            examples=[
                ["Who is the all-time leading scorer in NBA history?"],
                ["Tell me about the 1996 Bulls championship run"],
                ["Who won MVP in 2023?"],
                ["Compare LeBron James and Michael Jordan"]
            ],
        )
    
    return interface
