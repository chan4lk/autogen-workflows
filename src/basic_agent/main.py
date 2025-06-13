from autogen import ConversableAgent, LLMConfig
import os

def run():
    # Configure the LLM (we created this in the previous section)
    llm_config = LLMConfig(
        api_type="openai",
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.2
    )

    # Create a basic financial agent
    with llm_config:
        finance_agent = ConversableAgent(
            name="finance_agent",
            system_message="You are a financial assistant who helps analyze financial data and transactions."
        )

    # Run the agent with a prompt
    response = finance_agent.run(
        message="Can you explain what makes a transaction suspicious?",
        max_turns=1
    )

    # Iterate through the chat automatically with console output
    response.process()