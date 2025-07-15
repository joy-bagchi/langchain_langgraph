from typing import Annotated, Optional, Union, TypedDict, Literal

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, MessagesState, StateGraph


def my_reducer(
        left: list[str],
        right: Optional[Union[str, list[str]]],
) -> list[str]:
    """Reduce a list of numbers or strings into a single number or string."""
    if right:
        return left + ([right] if isinstance(right, str) else left) + right
    return left


class JobApplicationState(MessagesState):
    job_description: str
    is_suitable: bool
    application: str
    actions: Annotated[list[str], my_reducer]
    messages: Annotated[list[AnyMessage], add_messages]


def analyze_job_description(state: JobApplicationState) -> JobApplicationState:
    """Simulate analysis of job description."""
    state['is_suitable'] = 'Python' in state['job_description']
    return state


def generate_application(state: JobApplicationState) -> JobApplicationState:
    """Simulate generating a job application."""
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


def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    """Determine the next step based on suitability."""
    if state['is_suitable']:
        return "generate_application"
    return END


builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)
job_application_graph = builder.compile()



