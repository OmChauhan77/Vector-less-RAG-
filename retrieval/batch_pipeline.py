import argparse
import json
import os
import random
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import Indexing
from start import VectorlessAgent
from tracing_utils import setup_tracing, span


def _parse_sample_value(raw_value: str) -> int:
	match = re.search(r"(\d+)", raw_value)
	if not match:
		raise ValueError("Sample must be 1, 2, 3, 4, or 5.")
	value = int(match.group(1))
	if value not in {1, 2, 3, 4, 5}:
		raise ValueError("Sample must be 1, 2, 3, 4, or 5.")
	return value


def _load_queries(queries_path: str) -> List[Dict[str, Any]]:
	with open(queries_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	queries: List[Dict[str, Any]] = []
	for category, items in data.items():
		if not isinstance(items, list):
			continue
		for item in items:
			query = item.get("query") if isinstance(item, dict) else None
			if not query:
				continue
			queries.append(
				{
					"category": category,
					"query": query,
					"expected": item.get("answer") if isinstance(item, dict) else None,
				}
			)
	return random.sample(queries, min(len(queries), 20))  # Limit to 20 queries for testing


def _load_max_tokens_per_node() -> int:
	load_dotenv()
	max_tokens_raw = os.getenv("max_tokens_per_node", "20000")
	try:
		return int(max_tokens_raw)
	except ValueError as exc:
		raise ValueError("max_tokens_per_node must be an integer") from exc


def _index_markdown_file(path: str, base_dir: str, max_tokens: int) -> Tuple[str, str]:
	workspace_dir, registry_path, meta_path = Indexing._load_workspace(base_dir)
	metadata = Indexing._load_workspace_metadata(meta_path)
	registry = Indexing._load_registry(registry_path)
	abs_path = os.path.abspath(path)

	if abs_path in registry:
		output_id = registry[abs_path]
		output_path = metadata.get("outputs", {}).get(output_id, {}).get("output_path")
		if output_path:
			return output_id, output_path

	with open(abs_path, "r", encoding="utf-8") as f:
		markdown_text = f.read()

	root = Indexing.parse_markdown_to_nodes(markdown_text)
	Indexing._post_process_nodes(root, max_tokens)
	Indexing._summarize_nodes(root, max_words=200)
	cleaned_root = Indexing._strip_internal_keys(root)
	output_json = json.dumps(cleaned_root, indent=2, ensure_ascii=True)

	output_id = uuid.uuid4().hex
	output_name = f"{Indexing._safe_filename(os.path.basename(abs_path))}_{output_id}.json"
	output_path = os.path.join(workspace_dir, output_name)

	with open(output_path, "w", encoding="utf-8") as f:
		f.write(output_json)

	registry[abs_path] = output_id
	metadata.setdefault("outputs", {})[output_id] = {
		"input_path": abs_path,
		"output_path": output_path,
	}
	Indexing._save_registry(registry_path, registry)
	Indexing._save_workspace_metadata(meta_path, metadata)

	return output_id, output_path


def _collect_markdown_files(sample_dir: str) -> List[str]:
	md_files: List[str] = []
	for root, _, files in os.walk(sample_dir):
		for name in files:
			if name.lower().endswith(".md"):
				md_files.append(os.path.join(root, name))
	return sorted(md_files)


def run_pipeline(sample_value: str, base_dir: Optional[str] = None) -> str:
	base_dir = base_dir or os.getcwd()
	sample_num = _parse_sample_value(sample_value)
	sample_dir = os.path.join(base_dir, "data", f"sample {sample_num}")
	queries_path = os.path.join(base_dir, "test dataset", f"sample{sample_num}.json")
	setup_tracing("batch_pipeline")

	if not os.path.isdir(sample_dir):
		raise FileNotFoundError(f"Sample folder not found: {sample_dir}")
	if not os.path.exists(queries_path):
		raise FileNotFoundError(f"Query file not found: {queries_path}")

	queries = _load_queries(queries_path)
	if not queries:
		raise ValueError("No queries found in the test dataset.")

	markdown_files = _collect_markdown_files(sample_dir)
	if not markdown_files:
		raise ValueError("No markdown files found in the sample folder.")

	max_tokens = _load_max_tokens_per_node()
	results: Dict[str, Any] = {
		"sample": sample_num,
		"queries_count": len(queries),
		"documents_count": len(markdown_files),
		"documents": [],
		"queries": [],
	}

	indexed_docs: List[Dict[str, str]] = []
	for md_path in markdown_files:
		with span(
			"batch_pipeline.index_document",
			{"input_path": os.path.relpath(md_path, base_dir)},
		):
			output_id, output_path = _index_markdown_file(md_path, base_dir, max_tokens)
			indexed_docs.append(
				{
					"input_path": os.path.relpath(md_path, base_dir),
					"doc_id": output_id,
					"output_path": os.path.relpath(output_path, base_dir),
				}
			)

	results["documents"] = indexed_docs
	all_doc_ids = [doc["doc_id"] for doc in indexed_docs]
	for item in queries:
		with span(
			"batch_pipeline.query",
			{
				"category": item["category"],
				"query_length": str(len(item["query"])),
				"doc_count": str(len(all_doc_ids)),
			},
		):
			agent = VectorlessAgent(base_dir=base_dir)
			answer = agent.answer_query_across_docs(all_doc_ids, item["query"])
			results["queries"].append(
				{
					"category": item["category"],
					"query": item["query"],
					"expected": item["expected"],
					"answer": answer,
				}
			)

	output_dir = os.path.join(base_dir, "test dataset")
	output_path = os.path.join(output_dir, f"results_sample{sample_num}.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2, ensure_ascii=True)

	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Index a sample folder and run test queries against it."
	)
	parser.add_argument(
		"sample",
		help="Sample selection: 1, 2, 3, 4, or 5 (e.g. '1' or 'sample 1')",
	)
	args = parser.parse_args()

	output_path = run_pipeline(args.sample)
	print(f"Results saved to: {output_path}")


if __name__ == "__main__":
	main()
