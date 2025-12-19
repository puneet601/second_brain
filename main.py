from core.orchestrator import Orchestrator

from opentelemetry import trace
from otel_setup import setup_tracing

def main():
    print("Hello from your second-brain! The bot is ready to chat. Please reply with /exit to quit")
    setup_tracing()
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("orchestrator.run"):
        app = Orchestrator()
        app.run()



if __name__ == "__main__":
    main()
