import json
import os
import urllib.error
import urllib.request
from typing import Optional

from dotenv import load_dotenv

from tracing_utils import span


def summarize_text_with_openrouter(text: str, max_words: int = 200) -> str:
	"""Summarize text using an OpenRouter model configured via .env.

	Required .env keys:
	- OPENROUTER_API_KEY
	- OPENROUTER_BASE_URL (e.g., https://openrouter.ai/api/v1)
	- OPENROUTER_MODEL (e.g., google/gemini-2.0-flash)
	"""
	if not text or not text.strip():
		return ""

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

	endpoint = base_url.rstrip("/") + "/chat/completions"
	prompt = (
		"what information is contained in the following text? Summarize it in no more than "
		f"{max_words} words.\n\n"
		f"Text:\n{text.strip()}"
	)

	payload = {
		"model": model,
		"messages": [{"role": "user", "content": prompt}],
		"temperature": 0.2,
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

	with span("indexing.summarize_text", {"max_words": str(max_words)}):
		try:
			with urllib.request.urlopen(request, timeout=60) as response:
				body = response.read().decode("utf-8")
		except urllib.error.HTTPError as exc:
			error_body = exc.read().decode("utf-8", errors="ignore")
			raise RuntimeError(f"OpenRouter error {exc.code}: {error_body}") from exc
		except urllib.error.URLError as exc:
			raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

		data = json.loads(body)
		return _extract_summary_from_response(data)


def _extract_summary_from_response(data: dict) -> str:
	try:
		content = data["choices"][0]["message"]["content"]
		return content.strip()
	except (KeyError, IndexError, AttributeError) as exc:
		raise RuntimeError("Unexpected OpenRouter response format.") from exc
