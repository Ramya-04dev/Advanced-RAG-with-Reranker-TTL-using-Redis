🧠 Advanced RAG with Reranker & Redis TTL

📌 Overview

This project implements an Advanced Retrieval-Augmented Generation (RAG) system that improves the accuracy of AI-generated answers using:

- Semantic search
- Neural reranking
- Redis-based caching with TTL (Time-To-Live)

The system retrieves data from Wikipedia, processes it into embeddings, ranks relevance using a Cross-Encoder, and generates responses using a local LLM.

---

🚀 Features

- 🔍 Wikipedia-based information retrieval
- ⚡ FAISS vector database for fast semantic search
- 🧠 Cross-Encoder reranking for better accuracy
- 🤖 Local LLM (TinyLlama / Llama 3.2 via Ollama)
- 💾 Redis Semantic Cache with TTL
- 🌐 Streamlit interactive UI

---

🏗️ Architecture Flow

User Query
   ↓
Wikipedia Loader
   ↓
Text Splitting
   ↓
Embedding Generation
   ↓
FAISS Vector Search
   ↓
Top-K Retrieval
   ↓
Cross-Encoder Reranking
   ↓
Best Context Selection
   ↓
Prompt Engineering
   ↓
LLM (TinyLlama)
   ↓
Final Answer
   ↓
Redis Cache (TTL)

---

🛠️ Tech Stack

- Frontend: Streamlit
- LLM: Ollama (Llama 3.2 / TinyLlama)
- Embeddings: Sentence Transformers ("all-MiniLM-L6-v2")
- Vector DB: FAISS
- Reranker: Cross-Encoder ("ms-marco-MiniLM-L-6-v2")
- Cache: Redis Semantic Cache

---

📂 Project Structure

├── Reranker_web.py      # Main Streamlit app
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

---

⚙️ Installation

1. Clone Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create Virtual Environment (Optional)

python -m venv venv
venv\Scripts\activate   # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Install & Run Ollama

Make sure Ollama is installed and running:

ollama run llama3.2

5. Run Redis (Optional for Cache)

redis-server

---

▶️ Usage

Run the Streamlit app:

streamlit run Reranker_web.py

Steps:

1. Enter your query
2. System fetches Wikipedia data
3. FAISS retrieves relevant chunks
4. Cross-Encoder reranks results
5. LLM generates final answer
6. Redis caches response (optional)

---

💡 Example Query

What is Paracetamol?

---

📊 Key Concepts Used

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Neural Reranking
- Vector Databases
- Prompt Engineering
- Caching with TTL

---

⚠️ Notes

- Redis caching is optional (project runs without it)
- Ensure internet connection for Wikipedia data
- First run may take time due to model loading

---

👩‍💻 Author

N. Ramya

---

📜 License

This project is for educational purposes.
