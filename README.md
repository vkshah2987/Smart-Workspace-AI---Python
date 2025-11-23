# RAG FastAPI - Enterprise Document RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system built with FastAPI, featuring multi-user document management, hybrid search (dense + sparse), and cross-encoder reranking. Uses Google Gemini for embeddings and generation, FAISS for vector search, and MongoDB for document storage.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Application                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI (Port 8000)                           │
│  Endpoints: /upload, /query, /documents/{user_id}, /documents/{id}  │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌────────┐          ┌──────────┐         ┌──────────┐
    │ Redis  │          │  MongoDB │         │  Gemini  │
    │  RQ    │          │  Storage │         │   API    │
    └────────┘          └──────────┘         └──────────┘
         │                    │
         ▼                    │
    ┌────────┐                │
    │ Worker │                │
    │  RQ    │◄───────────────┘
    └────────┘
         │                    
         ▼                    
    ┌──────────┐         ┌────────────┐
    │  FAISS   │         │  Reranker  │
    │ Service  │         │  Service   │
    │Port 8001 │         │ Port 8501  │
    └──────────┘         └────────────┘
```

### Service Components

| Service | Port | Purpose | Technology |
|---------|------|---------|------------|
| **API** | 8000 | Main REST API | FastAPI + Uvicorn |
| **Worker** | - | Background document processing | RQ (Redis Queue) |
| **FAISS** | 8001 | Vector similarity search | FAISS + FastAPI |
| **Reranker** | 8501 | Cross-encoder reranking | Sentence-Transformers |
| **MongoDB** | 27017 | Document & chunk storage | MongoDB 6.0 |
| **Redis** | 6379 | Task queue | Redis 7 |

---

## 🚀 Features

### Core Functionality

- ✅ **Multi-user document management** - Isolated documents per user
- ✅ **Multi-format support** - PDF, DOCX, CSV, TXT
- ✅ **Async file upload** - Background processing with status tracking
- ✅ **Hybrid search** - Combines dense (FAISS) + sparse (MongoDB text search)
- ✅ **Cross-encoder reranking** - Re-scores candidates for better accuracy
- ✅ **Google Gemini integration** - Embeddings (`gemini-embedding-001`) + Generation (`gemini-2.5-flash`)
- ✅ **Document lifecycle management** - List, delete, status tracking (`queued` → `indexed`)
- ✅ **Persistent storage** - FAISS index + MongoDB with Docker volumes

### API Endpoints

#### 1. **POST /upload** - Upload Document
Upload a document for processing and indexing.

**Request:**
```bash
curl -X POST "http://localhost:8000/upload?user_id=vishal" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "doc_id": "6921cff43dcbf65b78dddc99",
  "location": "/app/uploads/2e11a573cca048409c8d8002f9fc6e9d.pdf"
}
```

**Process:**
1. File saved to disk with UUID filename
2. Document metadata stored in MongoDB (status: `"queued"`)
3. Background job queued in Redis
4. Worker processes: extract text → chunk → embed → store in FAISS
5. Status updated to `"indexed"` on completion

---

#### 2. **POST /query** - Query Documents
Ask questions and get AI-generated answers with source citations.

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "vishal",
    "query_text": "What is the main topic of the document?"
  }'
```

**Response:**
```json
{
  "answer": "The document discusses artificial intelligence applications in healthcare...",
  "sources": [
    {
      "doc_id": "6921cff43dcbf65b78dddc99",
      "chunk_id": "6921cff43dcbf65b78dddc99__0",
      "score": 0.87
    },
    {
      "doc_id": "6921cff43dcbf65b78dddc99",
      "chunk_id": "6921cff43dcbf65b78dddc99__5",
      "score": 0.82
    }
  ]
}
```

**Process:**
1. Query embedded using Google Gemini
2. Dense search via FAISS (top 10)
3. Sparse search via MongoDB text index (top 10)
4. Candidate deduplication (merge by chunk_id, keep highest score)
5. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
6. Top 3 chunks used as context
7. Google Gemini generates final answer

---

#### 3. **GET /documents/{user_id}** - List User Documents
Retrieve all documents for a specific user.

**Request:**
```bash
curl "http://localhost:8000/documents/vishal"
```

**Response:**
```json
{
  "user_id": "vishal",
  "documents": [
    {
      "doc_id": "6921cff43dcbf65b78dddc99",
      "filename": "secret.txt",
      "status": "indexed",
      "path": "/app/uploads/2e11a573cca048409c8d8002f9fc6e9d.txt"
    },
    {
      "doc_id": "6921cff43dcbf65b78dddc98",
      "filename": "report.pdf",
      "status": "queued",
      "path": "/app/uploads/3f22b684ddb159520d9d9003g0gd7f0e.pdf"
    }
  ]
}
```

---

#### 4. **DELETE /documents/{doc_id}** - Delete Document
Completely remove a document and all associated data.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/documents/6921cff43dcbf65b78dddc99"
```

**Response:**
```json
{
  "doc_id": "6921cff43dcbf65b78dddc99",
  "message": "Document and all associated data deleted successfully",
  "deleted": true
}
```

**Cleanup Process:**
1. Delete all chunks from MongoDB `chunks` collection
2. Remove vectors from FAISS index
3. Delete FAISS mappings from MongoDB `faiss_mappings` collection
4. Remove document from MongoDB `documents` collection
5. Delete physical file from disk

---

## 📁 Project Structure

```
rag-fastapi/
├── app/
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app & endpoints
│       ├── schemas.py           # Pydantic models
│       ├── storage.py           # File upload handler
│       └── clients/
│           ├── faiss_client.py      # FAISS service client
│           ├── gemini_client.py     # Google Gemini client
│           ├── mongo_client.py      # MongoDB wrapper
│           └── reranker_client.py   # Reranker service client
├── worker/
│   ├── __init__.py
│   ├── worker.py                # RQ worker job definitions
│   └── processors.py            # Document processing pipeline
├── faiss_service/
│   └── faiss_service.py         # FAISS vector search service
├── reranker/
│   └── reranker_service.py      # Cross-encoder reranking service
├── docker/
│   ├── Dockerfile.api           # API container
│   ├── Dockerfile.worker        # Worker container
│   ├── Dockerfile.faiss         # FAISS service container
│   └── Dockerfile.reranker      # Reranker service container
├── uploads/                     # Local file storage (mounted volume)
├── faiss_data/                  # FAISS index persistence (mounted volume)
├── docker-compose.yml           # Multi-service orchestration
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Python 3.11** - Runtime

### AI/ML
- **Google Gemini API** - Embeddings (`gemini-embedding-001`) + Generation (`gemini-2.5-flash`)
- **FAISS** (CPU) - Vector similarity search
- **Sentence-Transformers** - Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

### Storage
- **MongoDB 6.0** - Document & chunk storage with text search
- **Redis 7** - Task queue backend

### Background Processing
- **RQ (Redis Queue)** - Async job processing
- **python-docx** - DOCX parsing
- **pdfplumber** - PDF extraction
- **pandas** - CSV handling

### Infrastructure
- **Docker + Docker Compose** - Containerization & orchestration
- **Multi-volume persistence** - Separate volumes for uploads, FAISS index, and MongoDB

---

## ⚙️ Setup & Installation

### Prerequisites

- Docker & Docker Compose
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-fastapi
```

2. **Configure environment variables**

Edit `.env` file and add your Gemini API key:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

3. **Start all services**
```bash
docker-compose up --build
```

This will start:
- API server → http://localhost:8000
- FAISS service → http://localhost:8001
- Reranker service → http://localhost:8501
- MongoDB → localhost:27017
- Redis → localhost:6379
- RQ Worker (background)

4. **Verify services are running**
```bash
curl http://localhost:8000/docs  # FastAPI docs
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://mongo:27017` | MongoDB connection string |
| `MONGO_DB` | `ragdb` | MongoDB database name |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `FAISS_SERVICE_URL` | `http://faiss:8001` | FAISS service internal URL |
| `RERANKER_URL` | `http://reranker:8501/rerank` | Reranker service URL |
| `GEMINI_API_KEY` | **(required)** | Google Gemini API key |
| `UPLOAD_DIR` | `/app/uploads` | File upload directory |
| `EMBED_DIM` | Auto-detected | Embedding dimension (768 for Gemini) |

### Docker Compose Volumes

- **`mongo_data`** - MongoDB database persistence
- **`faiss_data`** - FAISS index file (`/data/faiss.index`)
- **`./uploads`** - Uploaded files (bind mount)

---

## 📊 Data Models

### MongoDB Collections

#### `documents`
```json
{
  "_id": ObjectId("6921cff43dcbf65b78dddc99"),
  "user_id": "vishal",
  "filename": "secret.txt",
  "path": "/app/uploads/2e11a573cca048409c8d8002f9fc6e9d.txt",
  "status": "indexed"  // "queued" | "indexed"
}
```

#### `chunks`
```json
{
  "chunk_id": "6921cff43dcbf65b78dddc99__0",
  "doc_id": "6921cff43dcbf65b78dddc99",
  "user_id": "vishal",
  "text": "This is a chunk of text from the document...",
  "seq": 0,
  "tokens": 127
}
```

#### `faiss_mappings`
```json
{
  "_id": 1234567890123456,  // SHA1 hash of chunk_id
  "chunk_id": "6921cff43dcbf65b78dddc99__0",
  "doc_id": "6921cff43dcbf65b78dddc99",
  "user_id": "vishal"
}
```

### Document Processing Pipeline

```
┌──────────────┐
│ Upload File  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Extract Text     │  ← PDF/DOCX/CSV/TXT
│ (processors.py)  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Chunk Text       │  ← 500 tokens, 100 stride
│ (chunk_text)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Store Chunks     │  → MongoDB chunks
│ (MongoDB)        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Generate         │  ← Google Gemini API
│ Embeddings       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Store Vectors    │  → FAISS index
│ (FAISS Service)  │  → MongoDB mappings
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Update Status    │  → status: "indexed"
│ (MongoDB)        │
└──────────────────┘
```

### Query Flow

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Embed Query      │  ← Google Gemini
└──────┬───────────┘
       │
       ├───────────────────────┐
       ▼                       ▼
┌──────────────┐     ┌──────────────────┐
│ Dense Search │     │ Sparse Search    │
│ (FAISS)      │     │ (MongoDB text)   │
│ Top 10       │     │ Top 10           │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       └───────────┬──────────┘
                   ▼
         ┌──────────────────┐
         │ Merge & Dedupe   │
         │ (by chunk_id)    │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Rerank           │  ← Cross-encoder
         │ (Top 3)          │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Generate Answer  │  ← Google Gemini
         │ with Sources     │
         └──────────────────┘
```

---

## 🧪 Testing & Usage Examples

### Upload Multiple Documents

```bash
# Upload PDF
curl -X POST "http://localhost:8000/upload?user_id=user1" \
  -F "file=@research_paper.pdf"

# Upload DOCX
curl -X POST "http://localhost:8000/upload?user_id=user1" \
  -F "file=@report.docx"

# Upload CSV
curl -X POST "http://localhost:8000/upload?user_id=user1" \
  -F "file=@data.csv"
```

### Query Across Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user1",
    "query_text": "Summarize the key findings from all uploaded documents"
  }'
```

### Monitor Document Status

```bash
# List all documents
curl "http://localhost:8000/documents/user1"

# Check if indexed (status should be "indexed", not "queued")
```

### Cleanup

```bash
# Delete specific document
curl -X DELETE "http://localhost:8000/documents/6921cff43dcbf65b78dddc99"
```

---

## 🐛 Troubleshooting

### Document Status Stuck at "queued"

**Symptoms:** Status never changes from `"queued"` to `"indexed"`

**Cause:** Worker unable to update MongoDB due to ObjectId mismatch (fixed in latest version)

**Solution:** Already fixed in `worker/worker.py` - converts string `doc_id` to `ObjectId` before updating

**Verify fix:**
```bash
# Check worker logs
docker-compose logs worker

# Should see successful processing without errors
```

### FAISS Index Dimension Mismatch

**Error:** `FAISS index dimension X mismatches embedding length Y`

**Solution:**
```bash
# Delete existing index and reingest
docker-compose down -v
docker-compose up --build
```

### MongoDB Text Search Fails

**Error:** `text index required for $text query`

**Solution:** Create text index manually
```bash
docker exec -it rag-fastapi-mongo-1 mongosh

use ragdb
db.chunks.createIndex({ text: "text" })
```

### Out of Memory (FAISS Service)

**Solution:** Switch to mmap-based FAISS index or increase Docker memory limit in `docker-compose.yml`:
```yaml
faiss:
  deploy:
    resources:
      limits:
        memory: 4G
```

---

## 🔒 Security Considerations

### Current Implementation
- ⚠️ No authentication/authorization
- ⚠️ User isolation based on `user_id` parameter (client-side trust)
- ⚠️ API key in `.env` file (not in version control)

### Production Recommendations
1. **Add JWT authentication** - Verify user identity
2. **Rate limiting** - Prevent abuse (use nginx or FastAPI middleware)
3. **Input validation** - Sanitize file uploads, limit file sizes
4. **Secrets management** - Use Docker secrets or HashiCorp Vault for `GEMINI_API_KEY`
5. **HTTPS/TLS** - Add reverse proxy (nginx, Traefik)
6. **Network isolation** - Internal services (FAISS, Reranker, MongoDB) not exposed externally

---

## 📈 Performance Optimization

### Current Configuration
- **Chunk size:** 500 tokens
- **Chunk overlap:** 100 tokens
- **Dense search:** Top 10 (FAISS)
- **Sparse search:** Top 10 (MongoDB)
- **Reranking:** Top 3 final candidates

### Tuning Recommendations

**For better accuracy:**
- Increase reranking candidates: `top_k = ranked[:5]`
- Use larger overlap: `CHUNK_STRIDE = 150`

**For faster queries:**
- Reduce search candidates: `top_k=5` in FAISS/MongoDB
- Cache embeddings for common queries
- Use FAISS IVF index for large datasets (>1M vectors)

**For scaling:**
- Add horizontal scaling for API/Worker services
- Use Redis Cluster for high-throughput job queues
- MongoDB replica set for read scaling
- FAISS GPU support (`faiss-gpu` instead of `faiss-cpu`)

---

## 🛣️ Roadmap & Future Enhancements

- [ ] **Authentication & Authorization** - JWT tokens, role-based access
- [ ] **Multi-tenancy** - Organization-level isolation
- [ ] **Advanced chunking** - Semantic chunking, sentence-aware boundaries
- [ ] **Metadata filtering** - Filter by date, file type, custom tags
- [ ] **Conversational memory** - Multi-turn dialogue support
- [ ] **Citation extraction** - Page numbers, paragraph references
- [ ] **Admin dashboard** - Monitor usage, document stats, user activity
- [ ] **Webhooks** - Notify when document processing completes
- [ ] **Export functionality** - Download processed chunks, embeddings
- [ ] **Alternative LLMs** - Support for OpenAI, Anthropic, local models
- [ ] **Vector database options** - Pinecone, Weaviate, Qdrant integration

---

## 📝 Development

### Local Development (without Docker)

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Start external services**
```bash
# MongoDB
docker run -d -p 27017:27017 mongo:6.0

# Redis
docker run -d -p 6379:6379 redis:7
```

3. **Update .env for local development**
```bash
MONGO_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379/0
FAISS_SERVICE_URL=http://localhost:8001
RERANKER_URL=http://localhost:8501/rerank
```

4. **Start services in separate terminals**
```bash
# Terminal 1: API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2: FAISS Service
uvicorn faiss_service.faiss_service:app --port 8001

# Terminal 3: Reranker Service
uvicorn reranker.reranker_service:app --port 8501

# Terminal 4: Worker
rq worker ingest
```

### Adding New Document Types

Edit `worker/processors.py`:
```python
def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".md":  # Example: Markdown support
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # ... existing code
```

### Custom Embedding Models

Edit `app/api/clients/gemini_client.py` or create new client in `app/api/clients/`.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Google Gemini** - Embedding and generation API
- **FAISS** - Facebook AI Similarity Search
- **Sentence-Transformers** - Cross-encoder models
- **FastAPI** - Modern Python web framework
- **RQ** - Simple Python job queues

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above

---

**Built with ❤️ using FastAPI, Google Gemini, and FAISS**
