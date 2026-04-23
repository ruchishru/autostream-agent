import json
import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ── Load knowledge base ──────────────────────────────────────────────
with open("knowledge_base.json", "r") as f:
    KNOWLEDGE_BASE = json.load(f)

KB_TEXT = f"""
AutoStream Pricing:
- Basic Plan: {KNOWLEDGE_BASE['plans']['basic']['price']} | {KNOWLEDGE_BASE['plans']['basic']['videos']} | {KNOWLEDGE_BASE['plans']['basic']['resolution']} resolution | Features: {', '.join(KNOWLEDGE_BASE['plans']['basic']['features'])}
- Pro Plan: {KNOWLEDGE_BASE['plans']['pro']['price']} | {KNOWLEDGE_BASE['plans']['pro']['videos']} | {KNOWLEDGE_BASE['plans']['pro']['resolution']} resolution | Features: {', '.join(KNOWLEDGE_BASE['plans']['pro']['features'])}

Company Policies:
- Refund Policy: {KNOWLEDGE_BASE['policies']['refund']}
- Support Policy: {KNOWLEDGE_BASE['policies']['support']}

About AutoStream: {KNOWLEDGE_BASE['company']['description']}
"""

# ── Mock lead capture tool ───────────────────────────────────────────
def mock_lead_capture(name: str, email: str, platform: str):
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured for {name} ({email}) on {platform}"

# ── LangGraph State ──────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    intent: str                  # greeting | inquiry | high_intent
    lead_name: str
    lead_email: str
    lead_platform: str
    lead_captured: bool
    collecting_lead: bool        # are we mid-collection?
    collection_step: str         # name | email | platform | done

# ── LLM setup ────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_retries=1,
)

SYSTEM_PROMPT = f"""You are AutoStream's helpful sales assistant.
Use ONLY the knowledge base below to answer product/pricing questions.

{KB_TEXT}

Intent Classification Rules:
- "greeting" → casual hello, no product interest
- "inquiry" → asking about features, pricing, policies
- "high_intent" → wants to sign up, try, buy, start, subscribe, get started

IMPORTANT:
- Never make up features or prices not in the knowledge base.
- When collecting lead info, ask for ONE piece at a time.
- Be friendly and concise.
"""

# ── Intent detection node ────────────────────────────────────────────
def detect_intent(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content.lower()

    high_intent_keywords = [
        "sign up", "subscribe", "buy", "purchase", "try", "get started",
        "want to", "i'm in", "sounds good", "let's go", "start",
        "i want", "give me", "enroll", "join"
    ]
    inquiry_keywords = [
        "price", "cost", "feature", "plan", "refund", "support",
        "how much", "what", "tell me", "explain", "difference", "compare"
    ]

    if any(kw in last_msg for kw in high_intent_keywords):
        intent = "high_intent"
    elif any(kw in last_msg for kw in inquiry_keywords):
        intent = "inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}

# ── Main response node ───────────────────────────────────────────────
def respond(state: AgentState) -> AgentState:
    # If already captured, just be friendly
    if state.get("lead_captured"):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {**state, "messages": [AIMessage(content=response.content)]}

    # Mid lead-collection flow
    if state.get("collecting_lead"):
        return handle_lead_collection(state)

    intent = state.get("intent", "greeting")

    if intent == "high_intent":
        # Start collecting lead info
        reply = (
            "That's awesome! I'd love to get you set up. 🎉\n\n"
            "Let's start — what's your **full name**?"
        )
        return {
            **state,
            "collecting_lead": True,
            "collection_step": "name",
            "messages": [AIMessage(content=reply)],
        }
    else:
        # RAG-powered response
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {**state, "messages": [AIMessage(content=response.content)]}

# ── Lead collection handler ──────────────────────────────────────────
def handle_lead_collection(state: AgentState) -> AgentState:
    step = state.get("collection_step", "name")
    last_input = state["messages"][-1].content.strip()

    if step == "name":
        return {
            **state,
            "lead_name": last_input,
            "collection_step": "email",
            "messages": [AIMessage(content=f"Nice to meet you, **{last_input}**! 👋\n\nWhat's your **email address**?")],
        }
    elif step == "email":
        # Basic email validation
        if "@" not in last_input or "." not in last_input:
            return {
                **state,
                "messages": [AIMessage(content="Hmm, that doesn't look like a valid email. Could you double-check it?")],
            }
        return {
            **state,
            "lead_email": last_input,
            "collection_step": "platform",
            "messages": [AIMessage(content="Got it! 📧\n\nWhich platform do you create content on? (e.g., YouTube, Instagram, TikTok, etc.)")],
        }
    elif step == "platform":
        lead_name = state.get("lead_name", "")
        lead_email = state.get("lead_email", "")

        # 🔥 Trigger the mock lead capture tool
        result = mock_lead_capture(lead_name, lead_email, last_input)

        reply = (
            f"🎉 You're all set, **{lead_name}**!\n\n"
            f"Our team will reach out to **{lead_email}** shortly to get your AutoStream Pro account activated.\n\n"
            f"Welcome aboard — can't wait to see your {last_input} content shine! 🚀"
        )
        return {
            **state,
            "lead_platform": last_input,
            "collection_step": "done",
            "collecting_lead": False,
            "lead_captured": True,
            "messages": [AIMessage(content=reply)],
        }

    return state

# ── Build LangGraph ──────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("respond", respond)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "respond")
    graph.add_edge("respond", END)

    return graph.compile()

app = build_graph()

# ── Public run function ──────────────────────────────────────────────
def run_agent(user_input: str, state: AgentState) -> tuple[str, AgentState]:
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
    result = app.invoke(state)
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "Sorry, I didn't get that."
    return reply, result
