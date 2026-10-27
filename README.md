# AI Volunteer Chatbot

A beginner-friendly Streamlit chatbot that uses retrieval-augmented generation
(RAG) to answer volunteer questions from the text files in `knowledge/`.
The chatbot is instructed not to invent information. If relevant information
cannot be retrieved, it says that the information is not available.

## Project structure

```text
AI_chatbot_project/
├── main.py                    # Streamlit user interface
├── chatbot.py                 # Grounded LLM answer generation
├── rag.py                     # Loading, chunking, embeddings, retrieval
├── config.py                  # Environment and application settings
├── requirements.txt           # Python dependencies
├── .env                       # Local Ollama and model settings
├── .gitignore                 # Ignores secrets and local files
├── README.md
└── knowledge/                 # Organization-provided source information
```

## Installation in VS Code on Windows

Open the project folder in VS Code and open **Terminal > New Terminal**.
Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in an ordinary PowerShell
window, then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install Ollama

Install Ollama for Windows from:

```text
https://ollama.com/download/windows
```

After installation, download the lightweight local chat model:

```powershell
ollama pull llama3.2:3b
```

The Ollama application must be running while you use the chatbot. Ollama
provides the local model through `http://127.0.0.1:11434` and does not require
an API key.

The `.env` file contains local settings:

```text
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Run the chatbot

```powershell
python -m streamlit run main.py
```

Streamlit will display a local URL. Open that URL in your browser. Stop the
server with `Ctrl+C`.

## How RAG works here

1. `rag.py` loads every `.txt` file from `knowledge/`.
2. Text is whitespace-cleaned and split into overlapping word chunks.
3. The local Sentence Transformers model represents each chunk as a numeric
   vector.
4. A question is embedded and compared with the chunks using cosine similarity.
5. Only chunks above `RELEVANCE_THRESHOLD` are passed to the local Ollama model.
6. `chatbot.py` asks Ollama to answer only from those chunks.
7. `main.py` displays the answer, source filenames, and session history.

Both embeddings and answer generation run locally. Embeddings use
`sentence-transformers/all-MiniLM-L6-v2`, and Ollama uses `llama3.2:3b` for
the final grounded answer. No cloud API or API key is used.

The knowledge files remain the source of truth. The included files contain
fictional academic demo information; replace it with approved information when
using the project for a real organization.

## Add or update knowledge

Edit the existing `.txt` files in `knowledge/`, keeping each topic in its
matching file. Do not add secrets. Restart Streamlit after changing files so
the retrieval index is rebuilt.

## Troubleshooting

- **Ollama connection error:** Start the Ollama application and confirm that
  `ollama list` shows `llama3.2:3b`.
- **Model not installed:** Run `ollama pull llama3.2:3b`.
- **No information available:** Add the approved answer to the relevant
  knowledge file, or ask a more specific question.
- **Knowledge folder/file error:** Confirm that `knowledge/` exists and contains
  readable, non-empty `.txt` files.
- **First embedding-model download:** The local sentence-transformers model is
  downloaded the first time it is used. Keep an internet connection available
  for that first download.
- **`streamlit` is not recognized:** Activate `.venv` and reinstall
  requirements with `python -m pip install -r requirements.txt`.
- **PowerShell activation error:** Use the `Set-ExecutionPolicy` command above.
