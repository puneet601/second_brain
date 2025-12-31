# observability.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

resource = Resource.create({
    "service.name": "second-brain",
    "service.version": "0.1.0"
})

trace.set_tracer_provider(
    TracerProvider(resource=resource)
)
metrics.set_meter_provider(
    MeterProvider(resource=resource)
)

trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(
        OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
    )
)



tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

request_latency = meter.create_histogram(
    name="request.latency.ms",
    unit="ms",
    description="End-to-end user request latency"
)

agent_latency = meter.create_histogram(
    name="agent.latency.ms",
    unit="ms",
    description="Latency per agent invocation"
)

agent_invocations = meter.create_counter(
    name="agent.invocations",
    description="Number of agent runs"
)

rag_hits = meter.create_counter(
    name="rag.hits",
    description="RAG retrievals performed"
)

memory_writes = meter.create_counter(
    name="memory.writes",
    description="Memory writes attempted"
)
