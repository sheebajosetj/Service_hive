import os
from typing import TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings



# -----------------------------
# CONFIG
# -----------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAUEA1ip6xhl1h5yrdLCQCWt5KtO-hdp60"

# LLM (Gemini is ONLY for text generation)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# Embeddings (LOCAL, no quota, no API calls)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# MOCK TOOL
# -----------------------------
def mock_lead_capture(name, email, platform):
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")

# -----------------------------
# STATE
# -----------------------------
class AgentState(TypedDict):
    messages: List[str]
    intent: str
    name: str
    email: str
    platform: str

# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------
def load_vectorstore():
    with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
        text = f.read()

    print("KB length:", len(text))

    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("Chunks:", len(chunks))
    print(
        "Sample chunk:",
        chunks[0].page_content[:200] if chunks else "NO CHUNKS"
    )

    return Chroma.from_documents(chunks, embeddings)


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# -----------------------------
# NODES
# -----------------------------
def classify_intent(state: AgentState):
    user_message = state["messages"][-1]

    prompt = f"""
Classify the user's intent into one of the following:
- casual_greeting
- product_or_pricing_inquiry
- high_intent_lead

User message: "{user_message}"

Respond with only the label.
"""

    intent = llm.invoke(prompt).content.strip()

    return {"intent": intent}

def handle_greeting(state: AgentState):
    return {"messages": state["messages"] + ["Hello! How can I help you with AutoStream today?"]}

def handle_rag(state: AgentState):
    query = state["messages"][-1]
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the user's question using only the information below.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt).content
    return {"messages": state["messages"] + [answer]}

def ask_for_details(state: AgentState):
    if not state.get("name"):
        return {"messages": state["messages"] + ["Great! May I have your name?"]}
    if not state.get("email"):
        return {"messages": state["messages"] + ["Thanks! Can you share your email address?"]}
    if not state.get("platform"):
        return {"messages": state["messages"] + ["Which creator platform do you use? (YouTube, Instagram, etc.)"]}

    mock_lead_capture(state["name"], state["email"], state["platform"])
    return {"messages": state["messages"] + ["You're all set! Our team will reach out soon."]}

def collect_details(state: AgentState):
    user_input = state["messages"][-1]

    if not state.get("name"):
        return {"name": user_input}
    if not state.get("email"):
        return {"email": user_input}
    if not state.get("platform"):
        return {"platform": user_input}

    return {}

# -----------------------------
# ROUTING
# -----------------------------
def route_intent(state: AgentState):
    if state["intent"] == "casual_greeting":
        return "greeting"
    if state["intent"] == "product_or_pricing_inquiry":
        return "rag"
    if state["intent"] == "high_intent_lead":
        return "lead"
    return END

# -----------------------------
# GRAPH
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("classify", classify_intent)
graph.add_node("greeting", handle_greeting)
graph.add_node("rag", handle_rag)
graph.add_node("lead", ask_for_details)
graph.add_node("collect", collect_details)

graph.set_entry_point("classify")

graph.add_conditional_edges("classify", route_intent)
graph.add_edge("greeting", END)
graph.add_edge("rag", END)
graph.add_edge("lead", "collect")
graph.add_edge("collect", "lead")

app = graph.compile()

# -----------------------------
# CLI LOOP
# -----------------------------
state = {
    "messages": [],
    "intent": "",
    "name": "",
    "email": "",
    "platform": ""
}

print("🤖 AutoStream Agent (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    state["messages"].append(user_input)
    state = app.invoke(state)
    print("Agent:", state["messages"][-1])
