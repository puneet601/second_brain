import os
import logging
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

def init_observability(local_only: bool = True) -> None:
    
    if local_only:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
        logger.info("OTel endpoint set to local otel-tui (localhost:4318)")
    else:
        logger.info("Using external OTel backend (Logfire or configured endpoint)")

    exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    set_tracer_provider(tracer_provider)

    Agent.instrument_all()

    logger.info("✅ Observability initialized successfully.")
