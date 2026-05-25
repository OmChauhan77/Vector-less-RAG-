import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set


def _load_workspace_metadata(base_dir: Optional[str] = None) -> Dict[str, Any]:
	base_dir = base_dir or os.getcwd()
	meta_path = os.path.join(base_dir, ".indexing_workspace.json")
	if not os.path.exists(meta_path):
		raise FileNotFoundError("Workspace metadata not found: .indexing_workspace.json")
	with open(meta_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _load_document_json(doc_id: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
	metadata = _load_workspace_metadata(base_dir)
	outputs = metadata.get("outputs", {})
	entry = outputs.get(doc_id)
	if not entry:
		raise ValueError(f"Unknown document ID: {doc_id}")
	output_path = entry.get("output_path")
	if not output_path or not os.path.exists(output_path):
		raise FileNotFoundError(f"Output JSON not found for ID {doc_id}: {output_path}")
	with open(output_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _remove_content(node: Dict[str, Any]) -> Dict[str, Any]:
	cleaned = {
		key: value
		for key, value in node.items()
		if key != "content"
	}
	children = node.get("children", [])
	if isinstance(children, list):
		cleaned["children"] = [_remove_content(child) for child in children]
	return cleaned


def get_structure(doc_id: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
	"""Return document JSON without the content field in each node."""
	doc = _load_document_json(doc_id, base_dir)
	if isinstance(doc, list):
		return [_remove_content(item) for item in doc]
	return _remove_content(doc)


def _collect_content_by_line_no(
	node: Dict[str, Any], targets: Set[int], results: Dict[int, str]
) -> None:
	line_no = node.get("line_no")
	if isinstance(line_no, int) and line_no in targets:
		results[line_no] = node.get("content", "")
	for child in node.get("children", []) or []:
		_collect_content_by_line_no(child, targets, results)


def get_content(doc_id: str, *line_nos: int, base_dir: Optional[str] = None) -> Dict[int, str]:
	"""Return content for nodes whose line_no is in the provided list."""
	if not line_nos:
		return {}
	targets = {int(value) for value in line_nos}
	results: Dict[int, str] = {}
	doc = _load_document_json(doc_id, base_dir)
	if isinstance(doc, list):
		for item in doc:
			_collect_content_by_line_no(item, targets, results)
	else:
		_collect_content_by_line_no(doc, targets, results)
	return results
