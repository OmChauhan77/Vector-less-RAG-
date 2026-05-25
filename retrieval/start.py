import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv

from retrieval import get_content, get_structure


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


def _call_openrouter(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
	api_key, base_url, model = _load_openrouter_config()
	endpoint = base_url.rstrip("/") + "/chat/completions"
	payload = {
		"model": model,
		"messages": messages,
		"temperature": temperature,
	}
	request = urllib.request.Request(
		endpoint,
		data=json.dumps(payload).encode("utf-8"),
		headers={
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
		},
		method="POST",
	)

	try:
		with urllib.request.urlopen(request, timeout=60) as response:
			body = response.read().decode("utf-8")
	except urllib.error.HTTPError as exc:
		error_body = exc.read().decode("utf-8", errors="ignore")
		raise RuntimeError(f"OpenRouter error {exc.code}: {error_body}") from exc
	except urllib.error.URLError as exc:
		raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

	data = json.loads(body)
	try:
		return data["choices"][0]["message"]["content"].strip()
	except (KeyError, IndexError, AttributeError) as exc:
		raise RuntimeError("Unexpected OpenRouter response format.") from exc


def _extract_line_nos(text: str) -> List[int]:
	try:
		data = json.loads(text)
		line_nos = data.get("line_nos", [])
		return [int(value) for value in line_nos if str(value).isdigit()]
	except json.JSONDecodeError:
		pass
	return [int(value) for value in re.findall(r"\b\d+\b", text)]


class VectorlessAgent:
	def __init__(self, base_dir: Optional[str] = None) -> None:
		self.base_dir = base_dir

	def _select_line_nos(self, doc_id: str, query: str) -> List[int]:
		structure = get_structure(doc_id, self.base_dir)
		messages = [
			{
				"role": "system",
				"content": (
					"You select relevant node line numbers from a document structure for a query. "
					"Return only JSON: {\"line_nos\":[...],\"rationale\":\"...\"}."
				),
			},
			{
				"role": "user",
				"content": (
					"Document structure (content removed):\n"
					f"{json.dumps(structure, ensure_ascii=True)}\n\n"
					f"Query: {query}\n"
					"Pick the smallest set of line_nos that likely contain the answer."
				),
			},
		]
		response = _call_openrouter(messages)
		line_nos = _extract_line_nos(response)
		return sorted(set(line_nos))

	def _answer_from_content(self, query: str, content_map: Dict[int, str]) -> str:
		content_lines = [
			f"Line {line_no}:\n{content}"
			for line_no, content in content_map.items()
		]
		context = "\n\n".join(content_lines)
		messages = [
			{
				"role": "system",
				"content": (
					"Answer the query using only the provided content. "
					"If the content is insufficient, say so explicitly."
				),
			},
			{
				"role": "user",
				"content": f"Query: {query}\n\nContent:\n{context}",
			},
		]
		return _call_openrouter(messages)

	def answer_query(self, doc_id: str, query: str) -> str:
		line_nos = self._select_line_nos(doc_id, query)
		if not line_nos:
			return "No relevant nodes found to answer the query."
		content_map = get_content(doc_id, *line_nos, base_dir=self.base_dir)
		if not content_map:
			return "No content found for the selected nodes."
		return self._answer_from_content(query, content_map)

	def answer_queries(self, doc_id: str, queries: Iterable[str]) -> Dict[str, str]:
		results: Dict[str, str] = {}
		for query in queries:
			results[query] = self.answer_query(doc_id, query)
		return results
