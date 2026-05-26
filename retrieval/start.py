import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openrouter import OpenRouter

from retrieval import get_content, get_structure
from tracing_utils import setup_tracing, span


def _load_openrouter_config() -> Tuple[str, str, str]:
	load_dotenv()
	api_key = os.getenv("OPENROUTER_API_KEY")
	base_url = os.getenv("OPENROUTER_BASE_URL")
	model = os.getenv("OPENROUTER_MODEL")
	if not api_key or not base_url or not model:
		missing = [
			key
			for key, value in {
				"OPENROUTER_API_KEY": api_key,
				"OPENROUTER_BASE_URL": base_url,
				"OPENROUTER_MODEL": model,
			}.items()
			if not value
		]
		raise ValueError(f"Missing required .env keys: {', '.join(missing)}")
	return api_key, base_url, model


def _build_agent(base_dir: Optional[str]) -> Agent:
	api_key, base_url, model_id = _load_openrouter_config()

	def get_structure_tool(doc_id: str) -> Dict[str, Any]:
		"""Return document structure without node content."""
		with span("tool.get_structure", {"doc_id": doc_id}):
			return get_structure(doc_id, base_dir)

	def get_content_tool(doc_id: str, line_nos: List[int]) -> Dict[int, str]:
		"""Return content for nodes whose line_no is in the provided list."""
		attrs = {
			"doc_id": doc_id,
			"line_count": str(len(line_nos)),
			"line_nos": ",".join(str(value) for value in line_nos),
		}
		with span("tool.get_content", attrs):
			return get_content(doc_id, *line_nos, base_dir=base_dir)

	instructions = [
		"You are a vector-less RAG assistant over indexed markdown nodes.",
		"When given multiple document IDs, call get_structure(doc_id) for each before selecting which document to use.",
		"Then call get_content(doc_id, line_nos=[...]) with relevant line numbers from the selected document(s).",
		"Iterate if needed and answer using only retrieved content.",
		"If content is insufficient, say so explicitly.",
		"Include cited line numbers and document IDs in the final answer.",
	]

	return Agent(
		model=OpenRouter(id=model_id, api_key=api_key, base_url=base_url),
		tools=[get_structure_tool, get_content_tool],
		instructions=instructions,
		markdown=True,
		tool_call_limit=12,
	)


class VectorlessAgent:
	def __init__(self, base_dir: Optional[str] = None) -> None:
		self.base_dir = base_dir
		setup_tracing("retrieval")
		self._agent = _build_agent(base_dir)

	def _attach_run_metrics(self, current_span, run_output) -> None:
		if current_span is None:
			return
		metrics = getattr(run_output, "metrics", None)
		if metrics is None:
			return
		for key in (
			"input_tokens",
			"output_tokens",
			"total_tokens",
			"prompt_tokens",
			"completion_tokens",
			"cost",
			"total_cost",
		):
			value = getattr(metrics, key, None)
			if value is not None:
				current_span.set_attribute(f"llm.{key}", str(value))
		details = getattr(metrics, "details", None)
		if isinstance(details, dict):
			for key, value in details.items():
				current_span.set_attribute(f"llm.details.{key}", str(value))

	def answer_query(self, doc_id: str, query: str) -> str:
		with span(
			"agent.run",
			{"doc_id": doc_id, "query_length": str(len(query))},
		) as current_span:
			prompt = (
				f"Document ID: {doc_id}\n"
				f"Question: {query}\n\n"
				"Follow the tool-use steps in your instructions."
			)
			run_output = self._agent.run(prompt)
			content = getattr(run_output, "content", None)
			if current_span is not None and content:
				current_span.set_attribute("llm.response_length", str(len(content)))
				current_span.set_attribute("llm.response_preview", content[:500])
			self._attach_run_metrics(current_span, run_output)
			return content if content is not None else str(run_output)

	def answer_query_across_docs(self, doc_ids: List[str], query: str) -> str:
		with span(
			"agent.run",
			{"doc_count": str(len(doc_ids)), "query_length": str(len(query))},
		) as current_span:
			doc_list = ", ".join(doc_ids)
			prompt = (
				f"Document IDs: {doc_list}\n"
				f"Question: {query}\n\n"
				"Call get_structure for each document ID before selecting which document(s) to use. "
				"Then call get_content for the relevant line numbers."
			)
			run_output = self._agent.run(prompt)
			content = getattr(run_output, "content", None)
			if current_span is not None and content:
				current_span.set_attribute("llm.response_length", str(len(content)))
				current_span.set_attribute("llm.response_preview", content[:500])
			self._attach_run_metrics(current_span, run_output)
			return content if content is not None else str(run_output)

	def answer_queries(self, doc_id: str, queries: Iterable[str]) -> Dict[str, str]:
		results: Dict[str, str] = {}
		for query in queries:
			results[query] = self.answer_query(doc_id, query)
		return results
