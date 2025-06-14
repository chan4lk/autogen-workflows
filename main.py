import sys
from dotenv import load_dotenv
load_dotenv()

from basic_agent import run
from research_agent import run_feedback_loop_pattern
from design_document_agent import run_design_document_agent
from hitl_agent import run_hitl_agent

if __name__ == "__main__":
    if sys.argv[1] == "basic":
        run()
    elif sys.argv[1] == "research":
        run_feedback_loop_pattern()
    elif sys.argv[1] == "design":
        run_design_document_agent()
    elif sys.argv[1] == "hitl":
        run_hitl_agent()
   
   
