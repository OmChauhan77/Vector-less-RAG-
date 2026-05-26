import argparse
import json
import os
import re
import uuid
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from Summary import summarize_text_with_openrouter
from tracing_utils import setup_tracing, span


Node = Dict[str, object]


ATX_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:#+\s*)?$")
SETEXT_UNDERLINE_RE = re.compile(r"^([=-]{2,})\s*$")


def _make_node(line_no: int, title: str, level: int) -> Node:
	return {
		"line_no": line_no,
		"title": title,
		"content": "",
		"children": [],
		"summary": "",
		"_level": level,
		"_content_lines": [],
	}


def _finalize_node(node: Node) -> None:
	lines = node.get("_content_lines", [])
	# Trim trailing blank lines but preserve internal spacing.
	while lines and lines[-1].strip() == "":
		lines.pop()
	node["content"] = "\n".join(lines)


def _try_parse_header(lines: List[str], index: int) -> Tuple[bool, int, str, int]:
	line = lines[index]
	match = ATX_HEADER_RE.match(line)
	if match:
		level = len(match.group(1))
		title = match.group(2).strip()
		return True, index, title, level

	if index + 1 < len(lines):
		next_line = lines[index + 1]
		underline = SETEXT_UNDERLINE_RE.match(next_line)
		if underline and line.strip() != "":
			level = 1 if underline.group(1).startswith("=") else 2
			title = line.strip()
			return True, index, title, level

	return False, index, "", 0


def parse_markdown_to_nodes(markdown_text: str) -> Node:
	lines = markdown_text.splitlines()
	root = _make_node(1, "ROOT", 0)
	stack: List[Node] = [root]

	index = 0
	while index < len(lines):
		is_header, header_line_index, title, level = _try_parse_header(lines, index)
		if is_header:
			_finalize_node(stack[-1])

			while stack and stack[-1]["_level"] >= level:
				stack.pop()
			if not stack:
				stack = [root]

			node = _make_node(header_line_index + 1, title, level)
			parent = stack[-1]
			parent["children"].append(node)
			stack.append(node)

			# Skip Setext underline line if used.
			if header_line_index + 1 < len(lines):
				next_line = lines[header_line_index + 1]
				if SETEXT_UNDERLINE_RE.match(next_line):
					index = header_line_index + 2
					continue

			index = header_line_index + 1
			continue

		stack[-1]["_content_lines"].append(lines[index])
		index += 1

	for node in stack:
		_finalize_node(node)
	_finalize_node(root)
	return root


def _strip_internal_keys(node: Node) -> Node:
	cleaned = {
		"line_no": node["line_no"],
		"title": node["title"],
		"content": node["content"],
		"children": [],
		"summary": node.get("summary", ""),
	}
	for child in node.get("children", []):
		cleaned["children"].append(_strip_internal_keys(child))
	return cleaned


def _load_workspace(base_dir: str) -> Tuple[str, str, str]:
	meta_path = os.path.join(base_dir, ".indexing_workspace.json")
	workspace_id = None
	metadata = {}
	if os.path.exists(meta_path):
		with open(meta_path, "r", encoding="utf-8") as f:
			metadata = json.load(f)
			workspace_id = metadata.get("workspace_id")
	if not workspace_id:
		workspace_id = uuid.uuid4().hex
		metadata = {"workspace_id": workspace_id, "outputs": {}}
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2)
	elif "outputs" not in metadata:
		metadata["outputs"] = {}
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2)

	workspace_dir = os.path.join(base_dir, f"index_workspace_{workspace_id}")
	os.makedirs(workspace_dir, exist_ok=True)
	registry_path = os.path.join(workspace_dir, "indexed_files.json")
	return workspace_dir, registry_path, meta_path


def _load_workspace_metadata(meta_path: str) -> Dict[str, object]:
	with open(meta_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _save_workspace_metadata(meta_path: str, metadata: Dict[str, object]) -> None:
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)


def _load_registry(registry_path: str) -> Dict[str, str]:
	if not os.path.exists(registry_path):
		return {}
	with open(registry_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _save_registry(registry_path: str, registry: Dict[str, str]) -> None:
	with open(registry_path, "w", encoding="utf-8") as f:
		json.dump(registry, f, indent=2)


def _safe_filename(name: str) -> str:
	base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
	return base or "indexed_output"


def _count_tokens(text: str) -> int:
	return len(re.findall(r"\S+", text))


def _alpha_suffix(index: int) -> str:
	index += 1
	letters = []
	while index > 0:
		index -= 1
		letters.append(chr(ord("a") + (index % 26)))
		index //= 26
	return "".join(reversed(letters))


def _split_by_page_break(text: str) -> List[str]:
	parts = re.split(r"\n\s*<!--\s*PageBreak\s*-->\s*\n", text)
	return [part for part in parts if part.strip()]


def _split_by_paragraph(text: str) -> List[str]:
	parts = re.split(r"\n\s*\n+", text)
	return [part for part in parts if part.strip()]


def _split_by_line(text: str) -> List[str]:
	parts = text.splitlines()
	return [part for part in parts if part.strip()]


def _build_line_records(node: Node) -> List[Tuple[int, str]]:
	lines = node.get("_content_lines") or node.get("content", "").splitlines()
	start_line = node["line_no"] + 1
	return [(start_line + index, line) for index, line in enumerate(lines)]


def _records_token_count(records: List[Tuple[int, str]]) -> int:
	return _count_tokens("\n".join(line for _, line in records))


def _split_records_by_page_break(
	records: List[Tuple[int, str]]
) -> List[List[Tuple[int, str]]]:
	chunks: List[List[Tuple[int, str]]] = [[]]
	for line_no, line in records:
		if re.search(r"<!--\s*PageBreak\s*-->", line):
			if chunks[-1]:
				chunks.append([])
			continue
		chunks[-1].append((line_no, line))
	return [chunk for chunk in chunks if chunk]


def _split_records_by_paragraph(
	records: List[Tuple[int, str]]
) -> List[List[Tuple[int, str]]]:
	chunks: List[List[Tuple[int, str]]] = [[]]
	for line_no, line in records:
		if line.strip() == "":
			if chunks[-1]:
				chunks.append([])
			continue
		chunks[-1].append((line_no, line))
	return [chunk for chunk in chunks if chunk]


def _split_records_by_line(
	records: List[Tuple[int, str]]
) -> List[List[Tuple[int, str]]]:
	return [[record] for record in records if record[1].strip()]


def _chunk_by_max_tokens(text: str, max_tokens: int) -> List[str]:
	words = re.findall(r"\S+", text)
	chunks: List[str] = []
	current: List[str] = []
	for word in words:
		current.append(word)
		if len(current) >= max_tokens:
			chunks.append(" ".join(current))
			current = []
	if current:
		chunks.append(" ".join(current))
	return chunks


def _split_text_to_chunks(text: str, max_tokens: int) -> List[str]:
	parts: List[str] = [text]
	for splitter in (_split_by_page_break, _split_by_paragraph, _split_by_line):
		new_parts: List[str] = []
		for part in parts:
			if _count_tokens(part) <= max_tokens:
				new_parts.append(part)
				continue
			sub_parts = splitter(part)
			if len(sub_parts) <= 1:
				new_parts.append(part)
			else:
				new_parts.extend(sub_parts)
		parts = new_parts

	final_parts: List[str] = []
	for part in parts:
		if _count_tokens(part) <= max_tokens:
			final_parts.append(part)
		else:
			final_parts.extend(_chunk_by_max_tokens(part, max_tokens))
	return [part for part in final_parts if part.strip()]


def _split_records_to_chunks(
	records: List[Tuple[int, str]], max_tokens: int
) -> List[List[Tuple[int, str]]]:
	parts: List[List[Tuple[int, str]]] = [records]
	for splitter in (
		_split_records_by_page_break,
		_split_records_by_paragraph,
		_split_records_by_line,
	):
		new_parts: List[List[Tuple[int, str]]] = []
		for part in parts:
			if _records_token_count(part) <= max_tokens:
				new_parts.append(part)
				continue
			sub_parts = splitter(part)
			if len(sub_parts) <= 1:
				new_parts.append(part)
			else:
				new_parts.extend(sub_parts)
		parts = new_parts

	final_parts: List[List[Tuple[int, str]]] = []
	for part in parts:
		if _records_token_count(part) <= max_tokens:
			final_parts.append(part)
			continue
		if len(part) == 1:
			line_no, line = part[0]
			word_chunks = _chunk_by_max_tokens(line, max_tokens)
			for chunk in word_chunks:
				final_parts.append([(line_no, chunk)])
		else:
			text = "\n".join(line for _, line in part)
			word_chunks = _chunk_by_max_tokens(text, max_tokens)
			for chunk in word_chunks:
				final_parts.append([(part[0][0], chunk)])
	return [part for part in final_parts if part]


def _attach_content_child_if_needed(node: Node) -> None:
	if node.get("children") and node.get("content", "").strip():
		line_records = _build_line_records(node)
		child_line_no = line_records[0][0] if line_records else node["line_no"]
		child = _make_node(child_line_no, f"{node['title']} (a)", node["_level"] + 1)
		child["content"] = node["content"]
		child["_content_lines"] = list(node.get("_content_lines", []))
		node["content"] = ""
		node["_content_lines"] = []
		node["children"].insert(0, child)


def _split_leaf_if_needed(node: Node, max_tokens: int) -> None:
	if node.get("children"):
		return
	content = node.get("content", "")
	if _count_tokens(content) <= max_tokens:
		return
	line_records = _build_line_records(node)
	if _records_token_count(line_records) <= max_tokens:
		return
	chunks = _split_records_to_chunks(line_records, max_tokens)
	if len(chunks) <= 1:
		return
	for index, chunk_records in enumerate(chunks):
		suffix = _alpha_suffix(index)
		chunk_lines = [line for _, line in chunk_records]
		child_line_no = chunk_records[0][0]
		child = _make_node(child_line_no, f"{node['title']} ({suffix})", node["_level"] + 1)
		child["_content_lines"] = chunk_lines
		_finalize_node(child)
		node["children"].append(child)
	node["content"] = ""
	node["_content_lines"] = []


def _post_process_nodes(node: Node, max_tokens: int) -> None:
	_attach_content_child_if_needed(node)
	_split_leaf_if_needed(node, max_tokens)
	for child in node.get("children", []):
		_post_process_nodes(child, max_tokens)


def _summarize_nodes(node: Node, max_words: int) -> None:
	for child in node.get("children", []):
		_summarize_nodes(child, max_words)

	if node.get("children"):
		child_summaries = [
			child.get("summary", "")
			for child in node["children"]
			if child.get("summary", "").strip()
		]
		combined = "\n".join(child_summaries).strip()
		node["summary"] = (
			summarize_text_with_openrouter(combined, max_words)
			if combined
			else ""
		)
	else:
		node["summary"] = summarize_text_with_openrouter(node["content"], max_words)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Index a Markdown file into hierarchical JSON nodes."
	)
	parser.add_argument("input", help="Path to a Markdown file")
	parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
	parser.add_argument(
		"--no-root",
		action="store_true",
		help="Emit a list of top-level nodes instead of a ROOT wrapper",
	)
	args = parser.parse_args()

	load_dotenv()
	setup_tracing("indexing")
	max_tokens_raw = os.getenv("max_tokens_per_node", "2000")
	try:
		max_tokens_per_node = int(max_tokens_raw)
	except ValueError:
		raise ValueError("max_tokens_per_node must be an integer")

	with open(args.input, "r", encoding="utf-8") as f:
		markdown_text = f.read()

	base_dir = os.getcwd()
	workspace_dir, registry_path, meta_path = _load_workspace(base_dir)
	metadata = _load_workspace_metadata(meta_path)
	registry = _load_registry(registry_path)
	input_path = os.path.abspath(args.input)
	if input_path in registry:
		output_id = registry[input_path]
		output_path = metadata.get("outputs", {}).get(output_id, {}).get("output_path")
		print(
			f"Skipping already indexed file: {input_path} (output: {output_path})"
		)
		return

	with span("indexing.index_file", {"input_path": input_path}):
		root = parse_markdown_to_nodes(markdown_text)
		_post_process_nodes(root, max_tokens_per_node)
		_summarize_nodes(root, max_words=200)
		cleaned_root = _strip_internal_keys(root)
		output_obj = cleaned_root["children"] if args.no_root else cleaned_root
		output_json = json.dumps(output_obj, indent=2, ensure_ascii=True)

	output_id = uuid.uuid4().hex
	output_name = f"{_safe_filename(os.path.basename(args.input))}_{output_id}.json"
	output_path = os.path.join(workspace_dir, output_name)

	with open(output_path, "w", encoding="utf-8") as f:
		f.write(output_json)
	registry[input_path] = output_id
	metadata.setdefault("outputs", {})[output_id] = {
		"input_path": input_path,
		"output_path": output_path,
	}
	_save_registry(registry_path, registry)
	_save_workspace_metadata(meta_path, metadata)

	if args.output:
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(output_json)
	else:
		print(output_json)


if __name__ == "__main__":
	main()
