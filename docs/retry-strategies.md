# Retry Strategies

Up to **2 retries** per receipt, strategy selected by error severity:

| Strategy | Trigger | Description |
|----------|---------|-------------|
| **LLM Self-Correction** | Small/medium errors | Send original text + error back to LLM with correction prompt |
| **RAG + Focused Context** | Medium/large errors | Extract only lines near errors, re-parse with focused context |
| **OCR Fallback** | Low OCR quality or large errors | Re-extract with OpenAI Vision, then re-parse |

Token usage is **accumulated across all attempts**.
