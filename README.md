# Payment Info Assistant

AI-powered assistant to query payment data using LLMs, vector search, and RAG pipelines. Includes:

✅ FastAPI backend for querying payment data & documents
✅ Streamlit frontend chat interface
✅ AWS Bedrock integration (Claude LLM & Embeddings)
✅ FAISS vector search for structured & unstructured data
✅ Synonym resolver and intelligent query handling

---

## Features

* 💬 Natural language Q\&A over payment transactions
* 📊 Combines structured data (Excel) with unstructured documents
* 🤖 Uses AWS Bedrock Claude for LLM responses
* 🔍 Vector search powered by FAISS
* 🛠️ Customizable prompts and business logic
* 📈 Logs, statistics, and admin tools included

---

## Project Structure

```
Payment-Info-Assistant/
├── app/               # FastAPI backend
├── UI/                # Streamlit frontend
├── utils/             # Utilities (logging, config, data search)
├── data/              # FAISS indexes, metadata, synonyms, chunks
├── logs/              # Log files
├── requirements.txt   # Python dependencies
├── dockerfile         # Docker container setup
└── README.md          # Project documentation
```

---

## Local Setup

### Prerequisites

* Python 3.11+
* AWS credentials with Bedrock access
* Docker (optional)

### 1. Clone & Install Dependencies

```bash
cd Payment-Info-Assistant
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AWS

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=eu-north-1  # Or your region
```

### 3. Prepare Data

Place required files in `/data`:

* `column_index.faiss`
* `column_metadata.pkl`
* `unstructured_index.faiss`
* `unstructured_chunks.pkl`
* `synonyms.json`

*(Contact admin if these are missing)*

### 4. Run Backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Run Frontend

```bash
streamlit run UI/streamlit_ui.py
```

Access UI at [http://localhost:8501](http://localhost:8501)

---

## Docker Setup (Optional)

```bash
docker build -t payment-assistant .
docker run -p 8000:8000 -p 8501:8501 payment-assistant
```

---

## API Endpoints

* `GET /health` - Health check
* `POST /ask` - Query payment data (`{"query": "your question"}`)
* `GET /logs` - View logs
* `GET /stats` - API usage statistics
* `GET /logs/download` - Download log file

---

## Frontend Features

* 💬 Chatbot interface for payment queries
* 📋 Logs viewer with filters
* 📊 API statistics dashboard
* 🔧 Settings & health monitoring

---

## Logs & Debugging

Logs stored in `/logs/logs.log`. Includes event types:

* `ERROR`, `SUCCESS`, `PROCESS`, `API_REQUEST`, `API_SUCCESS`, `API_ERROR`, etc.

---

## Requirements

Key dependencies:

* FastAPI, Uvicorn, Streamlit
* AWS SDK (boto3)
* faiss-cpu, numpy, pandas, sklearn
* transformers, torch (CPU-only)
* rapidfuzz, nltk

---

## Troubleshooting

* **AWS Errors**: Check credentials, region, Bedrock access
* **Data Missing**: Ensure `.faiss` and `.pkl` files in `/data`
* **Frontend Can't Connect**: Start backend on port 8000
* **CORS Issues**: Update `allow_origins` in `app/main.py`

---



