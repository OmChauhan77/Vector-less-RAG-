import argparse
import html
import json
import re
from typing import Dict, List, Tuple


Node = Dict[str, object]


ATX_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:#+\s*)?$")
SETEXT_UNDERLINE_RE = re.compile(r"^([=-]{2,})\s*$")
TABLE_SEPARATOR_RE = re.compile(
	r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
HTML_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)
HTML_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)
HTML_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
HTML_CELL_RE = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _make_node(line_no: int, title: str, level: int) -> Node:
	return {
		"line_no": line_no,
		"title": title,
		"content": "",
		"children": [],
		"tables": [],
		"_level": level,
		"_content_lines": [],
		"_table_count": 0,
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


def _split_table_row(line: str) -> List[str]:
	stripped = line.strip()
	if stripped.startswith("|"):
		stripped = stripped[1:]
	if stripped.endswith("|"):
		stripped = stripped[:-1]
	return [cell.strip() for cell in stripped.split("|")]


def _is_table_start(lines: List[str], index: int) -> bool:
	if index + 1 >= len(lines):
		return False
	if "|" not in lines[index]:
		return False
	return TABLE_SEPARATOR_RE.match(lines[index + 1]) is not None


def _is_html_table_start(line: str) -> bool:
	return HTML_TABLE_START_RE.search(line) is not None


def _strip_html(text: str) -> str:
	text = html.unescape(text)
	text = HTML_TAG_RE.sub("", text)
	return " ".join(text.split())


def _parse_html_table(lines: List[str], start_index: int) -> Tuple[Dict[str, object], int]:
	buffer = [lines[start_index]]
	index = start_index + 1
	while index < len(lines) and not HTML_TABLE_END_RE.search(lines[index]):
		buffer.append(lines[index])
		index += 1
	if index < len(lines):
		buffer.append(lines[index])
		index += 1

	table_html = "\n".join(buffer)
	rows: List[List[str]] = []
	columns: List[str] = []
	for row_html in HTML_TR_RE.findall(table_html):
		cells_raw = HTML_CELL_RE.findall(row_html)
		cells = [_strip_html(cell) for cell in cells_raw]
		if not cells:
			continue
		if not columns and re.search(r"<th\b", row_html, re.IGNORECASE):
			columns = cells
		else:
			rows.append(cells)

	if not columns:
		max_cols = max((len(row) for row in rows), default=0)
		columns = [f"column_{i + 1}" for i in range(max_cols)]

	return {"columns": columns, "rows": rows}, index


def _parse_table(lines: List[str], start_index: int) -> Tuple[Dict[str, object], int]:
	columns = _split_table_row(lines[start_index])
	rows: List[List[str]] = []
	index = start_index + 2
	while index < len(lines):
		line = lines[index]
		if line.strip() == "":
			break
			# Stop if a header starts at this line.
		if _try_parse_header(lines, index)[0]:
			break
		if "|" not in line:
			break
		if TABLE_SEPARATOR_RE.match(line):
			break
		rows.append(_split_table_row(line))
		index += 1
	return {"columns": columns, "rows": rows}, index


def parse_markdown_to_nodes(markdown_text: str) -> Node:
	lines = markdown_text.splitlines()
	root = _make_node(1, "ROOT", 0)
	stack: List[Node] = [root]
	in_code_block = False

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

		line = lines[index]
		stripped = line.lstrip()
		if stripped.startswith("```") or stripped.startswith("~~~"):
			in_code_block = not in_code_block
			stack[-1]["_content_lines"].append(line)
			index += 1
			continue

		if not in_code_block and _is_html_table_start(line):
			table, next_index = _parse_html_table(lines, index)
			table_count = stack[-1].get("_table_count", 0) + 1
			stack[-1]["_table_count"] = table_count
			table["name"] = f"table_{table_count}"
			stack[-1]["tables"].append(table)
			index = next_index
			continue

		if not in_code_block and _is_table_start(lines, index):
			table, next_index = _parse_table(lines, index)
			table_count = stack[-1].get("_table_count", 0) + 1
			stack[-1]["_table_count"] = table_count
			table["name"] = f"table_{table_count}"
			stack[-1]["tables"].append(table)
			index = next_index
			continue

		stack[-1]["_content_lines"].append(line)
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
		"tables": node.get("tables", []),
	}
	for child in node.get("children", []):
		cleaned["children"].append(_strip_internal_keys(child))
	return cleaned


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

	with open(args.input, "r", encoding="utf-8") as f:
		markdown_text = f.read()

	root = parse_markdown_to_nodes(markdown_text)
	cleaned_root = _strip_internal_keys(root)
	output_obj = cleaned_root["children"] if args.no_root else cleaned_root
	output_json = json.dumps(output_obj, indent=2, ensure_ascii=True)

	if args.output:
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(output_json)
	else:
		print(output_json)


if __name__ == "__main__":
	main()
