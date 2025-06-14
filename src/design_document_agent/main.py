from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
import os
import random

# Note: Make sure to set your API key in your environment first
def run_design_document_agent():
    # Configure the LLM
    llm_config = LLMConfig(
        api_type="openai",
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2,
    )

    # Define the system message for our architect agent
    architect_system_message = """
    you are an expert software architect. Generate a design document for software system.

    1. Ask for system requirements.
    2. Generate the outline of the document. Ask for approval or make changes based on the feedback.
    3. Generate the main content of the document. Ask for approval or make changes based on the feedback.
    4. Generate the conclusion of the document. Ask for approval or make changes based on the feedback.
    5. Generate the final document. Ask for approval or make changes based on the feedback.

    When all sections are processed, summarize the results and say "You can type exit to finish".
    
    """

    # Define the system message for our critic agent
    critic_system_message = """
    you are a critic. You will review the design document and provide feedback.
    """

    # Create the architect agent with LLM intelligence
    with llm_config:
        architect = ConversableAgent(
            name="architect",
            system_message=architect_system_message,
        )

        critic = ConversableAgent(
            name="critic",
            system_message=critic_system_message
        )

    # Create the human agent for oversight
    human = ConversableAgent(
        name="human",
        human_input_mode="ALWAYS",  # Always ask for human input
    )


    # Format the initial message
    initial_prompt = (
        "Please generate a design document for software system."
    )

    # Start the conversation from the human agent
    pattern = AutoPattern(
    initial_agent=architect,
    agents=[architect, critic],
    user_agent=human,
    group_manager_args = {"llm_config": llm_config},
)

    result, _, _ = initiate_group_chat(
        pattern=pattern,
        messages=initial_prompt,
    )