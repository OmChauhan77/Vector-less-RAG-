import os
from contextlib import contextmanager
from typing import Dict, Optional

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(service_name: str) -> Optional[trace.Tracer]:
	if getattr(setup_tracing, "_configured", False):
		return trace.get_tracer(service_name)

	load_dotenv()
	endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
	api_key = os.getenv("PHOENIX_API_KEY")
	if not endpoint or not api_key:
		return None

	resource = Resource.create({"service.name": service_name})
	provider = TracerProvider(resource=resource)
	exporter = OTLPSpanExporter(
		endpoint=endpoint,
		headers={"Authorization": f"Bearer {api_key}"},
	)
	provider.add_span_processor(BatchSpanProcessor(exporter))
	trace.set_tracer_provider(provider)
	setup_tracing._configured = True
	return trace.get_tracer(service_name)


@contextmanager
def span(name: str, attributes: Optional[Dict[str, str]] = None):
	tracer = trace.get_tracer(__name__)
	if not tracer:
		yield None
		return
	with tracer.start_as_current_span(name) as current_span:
		if attributes:
			for key, value in attributes.items():
				current_span.set_attribute(key, value)
		yield current_span
