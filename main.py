from agent import run_agent, AgentState

def main():
    print("\n" + "="*55)
    print("  🎬  Welcome to AutoStream AI Assistant  🎬")
    print("="*55)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    # Initialize empty state
    state: AgentState = {
        "messages": [],
        "intent": "greeting",
        "lead_name": "",
        "lead_email": "",
        "lead_platform": "",
        "lead_captured": False,
        "collecting_lead": False,
        "collection_step": "name",
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for chatting! Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("👋 Thanks for chatting with AutoStream! Goodbye.")
            break

        reply, state = run_agent(user_input, state)
        print(f"\nAutoStream Agent: {reply}\n")

if __name__ == "__main__":
    main()
