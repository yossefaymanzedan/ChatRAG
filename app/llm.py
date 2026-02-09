import json
from pathlib import Path
from typing import Any, Iterator

import httpx


class RuntimeLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        provider: str = "ollama",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
        openai_base_url: str = "https://api.openai.com/v1",
        deepseek_api_key: str = "",
        deepseek_model: str = "deepseek-chat",
        deepseek_base_url: str = "https://api.deepseek.com/v1",
        settings_path: str = ".rag/llm_settings.json",
    ) -> None:
        self.settings_path = Path(settings_path)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

        self.provider = (provider or "ollama").strip().lower()
        self.ollama_base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = (model or "").strip()
        self.temperature = float(temperature)

        self.openai_api_key = (openai_api_key or "").strip()
        self.openai_model = (openai_model or "gpt-4o-mini").strip()
        self.openai_base_url = (openai_base_url or "https://api.openai.com/v1").rstrip("/")

        self.deepseek_api_key = (deepseek_api_key or "").strip()
        self.deepseek_model = (deepseek_model or "deepseek-chat").strip()
        self.deepseek_base_url = (deepseek_base_url or "https://api.deepseek.com/v1").rstrip("/")

        self._load_runtime_config()

    def _load_runtime_config(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.provider = str(data.get("provider", self.provider) or self.provider).lower()
        self.ollama_base_url = str(data.get("ollama_base_url", self.ollama_base_url) or self.ollama_base_url).rstrip("/")
        self.ollama_model = str(data.get("ollama_model", self.ollama_model) or self.ollama_model)
        self.temperature = float(data.get("temperature", self.temperature) or self.temperature)

        self.openai_api_key = str(data.get("openai_api_key", self.openai_api_key) or self.openai_api_key)
        self.openai_model = str(data.get("openai_model", self.openai_model) or self.openai_model)
        self.openai_base_url = str(data.get("openai_base_url", self.openai_base_url) or self.openai_base_url).rstrip("/")

        self.deepseek_api_key = str(data.get("deepseek_api_key", self.deepseek_api_key) or self.deepseek_api_key)
        self.deepseek_model = str(data.get("deepseek_model", self.deepseek_model) or self.deepseek_model)
        self.deepseek_base_url = str(data.get("deepseek_base_url", self.deepseek_base_url) or self.deepseek_base_url).rstrip("/")

    def _save_runtime_config(self) -> None:
        payload = {
            "provider": self.provider,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "temperature": self.temperature,
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
            "openai_base_url": self.openai_base_url,
            "deepseek_api_key": self.deepseek_api_key,
            "deepseek_model": self.deepseek_model,
            "deepseek_base_url": self.deepseek_base_url,
        }
        self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_runtime_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "temperature": self.temperature,
            "ollama": {
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            },
            "openai": {
                "base_url": self.openai_base_url,
                "model": self.openai_model,
                "has_api_key": bool(self.openai_api_key),
            },
            "deepseek": {
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model,
                "has_api_key": bool(self.deepseek_api_key),
            },
        }

    def update_runtime_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(payload.get("provider", self.provider) or self.provider).strip().lower()
        if provider not in {"ollama", "openai", "deepseek"}:
            raise ValueError("provider must be one of: ollama, openai, deepseek")

        self.provider = provider

        if "temperature" in payload:
            self.temperature = float(payload.get("temperature") or self.temperature)

        ollama = payload.get("ollama") or {}
        if "base_url" in ollama:
            self.ollama_base_url = str(ollama.get("base_url") or self.ollama_base_url).rstrip("/")
        if "model" in ollama:
            self.ollama_model = str(ollama.get("model") or self.ollama_model)

        openai = payload.get("openai") or {}
        if "base_url" in openai:
            self.openai_base_url = str(openai.get("base_url") or self.openai_base_url).rstrip("/")
        if "model" in openai:
            self.openai_model = str(openai.get("model") or self.openai_model)
        if "api_key" in openai:
            self.openai_api_key = str(openai.get("api_key") or "").strip()

        deepseek = payload.get("deepseek") or {}
        if "base_url" in deepseek:
            self.deepseek_base_url = str(deepseek.get("base_url") or self.deepseek_base_url).rstrip("/")
        if "model" in deepseek:
            self.deepseek_model = str(deepseek.get("model") or self.deepseek_model)
        if "api_key" in deepseek:
            self.deepseek_api_key = str(deepseek.get("api_key") or "").strip()

        self._save_runtime_config()
        return self.get_runtime_config()

    def is_configured(self) -> bool:
        if self.provider == "ollama":
            return bool(self.ollama_model)
        if self.provider == "openai":
            return bool(self.openai_model and self.openai_api_key)
        if self.provider == "deepseek":
            return bool(self.deepseek_model and self.deepseek_api_key)
        return False

    def _chat_payload(self, system_prompt: str, user_prompt: str, stream: bool) -> dict[str, Any]:
        return {
            "model": self.ollama_model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
            },
        }

    @staticmethod
    def _parse_json_text(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _openai_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _chat_json_openai_compatible(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        "Return ONLY valid JSON. No markdown, no explanation, no backticks."
                    ),
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers=self._openai_headers(api_key),
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        content = str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
        return self._parse_json_text(content)

    def _chat_stream_openai_compatible(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "stream": True,
        }
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=self._openai_headers(api_key),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        event = json.loads(chunk)
                    except Exception:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield str(delta)

    def chat_json(self, system_prompt: str, user_prompt: str, timeout: float = 90.0) -> dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("LLM provider is not configured. Open Settings and set provider/model/API key.")

        if self.provider == "ollama":
            payload = self._chat_payload(system_prompt=system_prompt, user_prompt=user_prompt, stream=False)
            payload["format"] = "json"
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama request failed. Is Ollama running and model '{self.ollama_model}' available?"
                ) from exc
            data = resp.json()
            content = str((data.get("message") or {}).get("content") or "").strip()
            return self._parse_json_text(content)

        if self.provider == "openai":
            try:
                return self._chat_json_openai_compatible(
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                raise RuntimeError("OpenAI request failed. Check API key/model/base URL.") from exc

        if self.provider == "deepseek":
            try:
                return self._chat_json_openai_compatible(
                    base_url=self.deepseek_base_url,
                    api_key=self.deepseek_api_key,
                    model=self.deepseek_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                raise RuntimeError("DeepSeek request failed. Check API key/model/base URL.") from exc

        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def chat_stream_markdown(self, system_prompt: str, user_prompt: str, timeout: float = 180.0):
        if not self.is_configured():
            raise RuntimeError("LLM provider is not configured. Open Settings and set provider/model/API key.")

        if self.provider == "ollama":
            payload = self._chat_payload(system_prompt=system_prompt, user_prompt=user_prompt, stream=True)
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except Exception:
                                continue
                            chunk = str((event.get("message") or {}).get("content") or "")
                            if chunk:
                                yield chunk
                            if bool(event.get("done")):
                                break
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama stream failed. Is Ollama running and model '{self.ollama_model}' available?"
                ) from exc
            return

        if self.provider == "openai":
            try:
                yield from self._chat_stream_openai_compatible(
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                raise RuntimeError("OpenAI stream failed. Check API key/model/base URL.") from exc
            return

        if self.provider == "deepseek":
            try:
                yield from self._chat_stream_openai_compatible(
                    base_url=self.deepseek_base_url,
                    api_key=self.deepseek_api_key,
                    model=self.deepseek_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                raise RuntimeError("DeepSeek stream failed. Check API key/model/base URL.") from exc
            return

        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def ollama_list_models(self, timeout: float = 20.0) -> list[dict[str, Any]]:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{self.ollama_base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") or []
        out: list[dict[str, Any]] = []
        for m in models:
            out.append(
                {
                    "name": str(m.get("name") or ""),
                    "size": m.get("size"),
                    "modified_at": m.get("modified_at"),
                }
            )
        return out

    def ollama_pull_model(self, model_name: str, timeout: float = 900.0):
        payload = {"name": model_name, "stream": True}
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self.ollama_base_url}/api/pull", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    total = ev.get("total")
                    completed = ev.get("completed")
                    pct = None
                    if isinstance(total, int) and total > 0 and isinstance(completed, int):
                        pct = round((completed / total) * 100.0, 2)
                    yield {
                        "status": ev.get("status") or "pulling",
                        "digest": ev.get("digest"),
                        "total": total,
                        "completed": completed,
                        "percent": pct,
                        "done": bool(ev.get("status") == "success"),
                    }


# Backward-compatible alias for existing imports.
DeepSeekClient = RuntimeLLMClient
