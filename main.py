from dotenv import load_dotenv
load_dotenv()

from basic_agent import run
from research_agent import run_feedback_loop_pattern

if __name__ == "__main__":
    run_feedback_loop_pattern()
    
