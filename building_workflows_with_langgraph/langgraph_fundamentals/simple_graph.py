from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from IPython.display import display, Image


class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str

def analyze_job_description(state: JobApplicationState) -> JobApplicationState:
    # Simulate analysis of job description
    state['is_suitable'] = 'Python' in state['job_description']
    return state

def generate_application(state: JobApplicationState) -> JobApplicationState:
    # Simulate generating a job application
    if state['is_suitable']:
        state['application'] = (
            "Dear Hiring Manager,\n\n"
            "I am excited to apply for the position as I have extensive experience in Python development.\n\n"
            "Best regards,\n"
            "Applicant"
        )
    else:
        state['application'] = (
            "Dear Hiring Manager,\n\n"
            "Thank you for considering my application, but I do not meet the requirements for this position.\n\n"
            "Best regards,\n"
            "Applicant"
        )

    return state

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application", END)
job_application_graph = builder.compile()




