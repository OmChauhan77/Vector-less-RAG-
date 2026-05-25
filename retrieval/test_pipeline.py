from start import VectorlessAgent


def main() -> None:
	doc_id = input("Document ID: ").strip()
	if not doc_id:
		print("Document ID is required.")
		return

	queries = []
	print("Enter queries (blank line to finish):")
	while True:
		query = input("> ").strip()
		if not query:
			break
		queries.append(query)

	if not queries:
		print("At least one query is required.")
		return

	agent = VectorlessAgent()
	results = agent.answer_queries(doc_id, queries)

	for query, answer in results.items():
		print("=" * 80)
		print(f"Q: {query}")
		print("A:")
		print(answer)


if __name__ == "__main__":
	main()
