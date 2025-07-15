from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
from langgraph.constants import Send
import operator

llm = OpenAI()

class AgentState(TypedDict):
    video_uri: str
    chunks: int
    interval_secs: int
    summaries: Annotated[list, operator.add]  # type: ignore
    final_summary: str


class _ChunkState(TypedDict):
    video_uri: str
    start_offset: int
    interval_secs: int

human_part = {'type': 'text', 'text': 'Provide a summary of the video.'}

async def summarize_video(state: _ChunkState):
    start_offset = state['start_offset']
    interval_secs = state['interval_secs']
    video_part = {
        'type': 'media', 'file_uri': state['video_uri'], 'mime_type': 'video/mp4',
        'video_metadata':
            {
                'start_offset': {'seconds': start_offset*interval_secs},
                'end_offset': {'seconds': (start_offset+1)*interval_secs}
            }
    }
    response = await llm.ainvoke(
        [HumanMessage(content=[human_part, video_part])]
    )
    return {"summaries": [response]}

async def _generate_final_summary(state: AgentState):
    summaries = state['summaries']
    if not summaries:
        return {"final_summary": "No summaries available."}

    final_summary = " ".join(summaries)
    return {"final_summary": final_summary}