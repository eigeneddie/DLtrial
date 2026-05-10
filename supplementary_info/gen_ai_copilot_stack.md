# GenAI Copilot Stack

This repository includes a local GenAI copilot for the Chilli Chiplets thermal co-design app.

## What The Copilot Does

- Reads the live structured thermal design state from the Streamlit app.
- Uses the current chiplet layout, power map, substrate, TSV cooling settings, peak temperature, hotspots, CTE stress, and AI-vs-FDM validation metrics.
- Retrieves relevant thermal-packaging context from a local Qdrant vector database.
- Sends the structured state plus retrieved context to a local Ollama model.
- Returns a short engineering recommendation for what to change first.

## Open-Source Stack

- **Ollama**: local LLM runtime.
- **Gemma (`gemma4:e4b`)**: local chat model used for the copilot answer.
- **Qdrant**: local vector database for thermal-packaging knowledge.
- **Nomic Embed Text (`nomic-embed-text`)**: local embedding model used for retrieval.
- **LangChain Community**: embedding integration for Ollama.
- **Streamlit**: app interface.

## Key Files

- `app.py`: main app and UI. The `Ask Copilot` button calls the local RAG stack.
- `rag/rag_engine.py`: retrieval plus Ollama/Gemma answer generation.
- `rag/config.py`: model names, local paths, prompt, and retrieval settings.
- `rag/ingest_thermal_papers.py`: ingestion script for converting source documents into Qdrant chunks.
- `rag/qdrant_storage/`: local demo vector index.

## Demo Flow

1. Start Ollama.
2. Run the Streamlit app.
3. Place chiplets or use the default layout.
4. Click **Run Analysis**.
5. Ask: `What does the heat map indicate and what should I change first?`
6. Click **Ask Copilot**.

Expected behavior: the copilot answers with a concise interpretation of the current heat map and one first design action, such as changing TSV strip count, TSV orientation, substrate material, or component placement.

## Notes For Judges

The copilot is designed to be local-first and open-source. It does not require sending the thermal design state to a cloud LLM for the main `Ask Copilot` flow. Optional online evidence search can be configured with a Perplexity API key, but the core demo uses the local Gemma + Qdrant stack.
