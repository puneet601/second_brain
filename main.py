from core.observability import init_observability
from core.orchestrator import Orchestrator


def main():
    print("Hello from your second-brain! The bot is ready to chat. Please reply with /exit to quit")
    query = input("You:")



if __name__ == "__main__":
    init_observability(local_only=True)
    app = Orchestrator()
    app.run()
