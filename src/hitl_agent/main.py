from autogen import ConversableAgent, LLMConfig
import os
import random

# Note: Make sure to set your API key in your environment first
def run_hitl_agent():
    # Configure the LLM
    llm_config = LLMConfig(
        api_type="openai",
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.2,
    )

    # Define the system message for our finance bot
    finance_system_message = """
    You are a financial compliance assistant. You will be given a set of transaction descriptions.
    For each transaction:
    - If it seems suspicious (e.g., amount > $10,000, vendor is unusual, memo is vague), ask the human agent for approval.
    - Otherwise, approve it automatically.
    Provide the full set of transactions to approve at one time.
    If the human gives a general approval, it applies to all transactions requiring approval.
    When all transactions are processed, summarize the results and say "You can type exit to finish".
    """

    # Create the finance agent with LLM intelligence
    with llm_config:
        finance_bot = ConversableAgent(
            name="finance_bot",
            system_message=finance_system_message,
        )

    # Create the human agent for oversight
    human = ConversableAgent(
        name="human",
        human_input_mode="ALWAYS",  # Always ask for human input
    )

    # Generate sample transactions - this creates different transactions each time you run
    VENDORS = ["Staples", "Acme Corp", "CyberSins Ltd", "Initech", "Globex", "Unicorn LLC"]
    MEMOS = ["Quarterly supplies", "Confidential", "NDA services", "Routine payment", "Urgent request", "Reimbursement"]

    def generate_transaction():
        amount = random.choice([500, 1500, 9999, 12000, 23000, 4000])
        vendor = random.choice(VENDORS)
        memo = random.choice(MEMOS)
        return f"Transaction: ${amount} to {vendor}. Memo: {memo}."

    # Generate 3 random transactions
    transactions = [generate_transaction() for _ in range(3)]

    # Format the initial message
    initial_prompt = (
        "Please process the following transactions one at a time:\n\n" +
        "\n".join([f"{i+1}. {tx}" for i, tx in enumerate(transactions)])
    )

    # Start the conversation from the human agent
    response = human.run(
        recipient=finance_bot,
        message=initial_prompt,
    )

    # Display the response
    response.process()