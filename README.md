# 🎬 AutoStream AI Sales Agent

A conversational AI agent for **AutoStream** — a fictional SaaS video editing platform — built as part of the **ServiceHive / Inflx Machine Learning Intern Assignment**.

The agent handles intent detection, RAG-powered product Q&A, and lead capture using **LangGraph** + **Groq (Llama 3.3 70B)** — completely free, no credit card required.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com) (sign up with Google, no credit card needed)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

```bash
pip install langgraph langchain langchain-google-genai python-dotenv langchain-groq
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Agent

```bash
python3 main.py
```

### 6. Try This Test Conversation

```
You: Hi there!
You: What plans do you offer?
You: What's included in the Pro plan?
You: That sounds great, I want to try the Pro plan for my YouTube channel
You: John Smith
You: john@example.com
You: YouTube
```

You should see a `✅ Lead captured successfully!` message after the last step.

---

## 🏗️ Architecture (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides an explicit, inspectable state machine — ideal for a multi-step lead qualification workflow where precise control over transitions (greeting → inquiry → lead collection → tool execution) is critical. AutoGen excels at multi-agent debates and collaboration; LangGraph is purpose-built for single-agent workflows with deterministic branching logic and stateful memory, making it the right fit here.

### How State is Managed

The agent uses a `TypedDict`-based `AgentState` that persists across all conversation turns within a session. LangGraph's `add_messages` annotation accumulates the full conversation history, giving the LLM complete context on every invocation. Custom state fields — `intent`, `collecting_lead`, `collection_step`, `lead_name`, `lead_email`, `lead_platform`, and `lead_captured` — act as a lightweight state machine. On every turn, the graph checks these fields to decide whether to answer a product question, advance lead collection, or fire the mock lead capture tool. This design prevents premature tool calls: the tool only triggers after all three required values (name, email, platform) are collected and validated sequentially.

### RAG Pipeline

The knowledge base is stored in a local `knowledge_base.json` file and loaded at startup, then injected into the LLM system prompt on every call. This is a simple but effective retrieval pattern for small, static knowledge bases. For production use, this would be replaced with a vector store (e.g., ChromaDB with sentence-transformer embeddings) for scalable semantic search.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

**1. Meta Developer Setup**
- Create an app at [developers.facebook.com](https://developers.facebook.com)
- Enable the **WhatsApp Business API** and get a phone number

**2. Wrap Agent in a Web Server**

```python
from fastapi import FastAPI, Request
from agent import run_agent, AgentState

app = FastAPI()
sessions = {}  # keyed by sender phone number

def default_state() -> AgentState:
    return {
        "messages": [], "intent": "greeting",
        "lead_name": "", "lead_email": "", "lead_platform": "",
        "lead_captured": False, "collecting_lead": False, "collection_step": "name"
    }

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    msg = body["entry"][0]["changes"][0]["value"]["messages"][0]
    sender_id = msg["from"]
    user_text = msg["text"]["body"]

    reply, sessions[sender_id] = run_agent(user_text, sessions.get(sender_id, default_state()))
    send_whatsapp_message(sender_id, reply)  # use Meta Graph API
    return {"status": "ok"}

@app.get("/webhook")
async def verify(request: Request):
    # Handle Meta's verification challenge
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "my_verify_token":
        return int(params["hub.challenge"])
```

**3. Session Persistence**
Use a dictionary keyed by `sender_id` (phone number) for development. For production, store sessions in **Redis** with a TTL to handle concurrent users and server restarts.

**4. Deploy**
Host on [Railway](https://railway.app), [Render](https://render.com), or any platform that provides a public HTTPS URL — required by Meta for webhook verification.

**5. Register Webhook**
In the Meta Developer portal, register your `/webhook` URL and subscribe to the `messages` event.

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py              # LangGraph agent — intent detection, RAG, tool calling
├── main.py               # CLI conversation loop
├── knowledge_base.json   # AutoStream pricing & policies (RAG source)
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed to git)
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Agent Framework | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| RAG Source | Local JSON knowledge base |
| State Management | LangGraph `TypedDict` state |
| Environment | python-dotenv |

---

## ✅ Evaluation Checklist

- ✅ Intent detection — greeting / inquiry / high_intent
- ✅ RAG from local knowledge base (pricing, features, policies)
- ✅ State management across 5–6 conversation turns
- ✅ Sequential lead collection — name → email → platform
- ✅ Tool fires **only** after all 3 values are collected and validated
- ✅ Free LLM — no credit card required (Groq free tier)
- ✅ Clean, modular code structure
- ✅ WhatsApp deployment plan documented

---

## 📝 Notes

- The `.env` file is excluded from git via `.gitignore` — never commit API keys
- The `mock_lead_capture()` function prints to terminal; in production this would POST to a CRM API (HubSpot, Salesforce, etc.)
- Basic email validation is included (checks for `@` and `.`)
