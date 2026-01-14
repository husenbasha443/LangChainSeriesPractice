from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Annotated,List
from langgraph.graph import add_messages
from IPython.display import Image,display
from langgraph.graph import StateGraph,START,END

load_dotenv()

model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq"
)

class State(TypedDict):
    messages:Annotated[list,add_messages]

def superbot(state:State):
    return {"messages":[model.invoke(state["messages"])]}


    ## Define the Graph
graph = StateGraph(State)

## Define the nodes
graph.add_node("superbot",superbot)

## Define the edges
graph.add_edge(START,"superbot")
graph.add_edge("superbot",END)

## Compile the Graph
graph_builder = graph.compile()

# Draw the Graph
display(Image(graph_builder.get_graph().draw_mermaid_png()))


response = graph_builder.invoke({"messages":"What is Artificial Intelligence"})
print(response)

