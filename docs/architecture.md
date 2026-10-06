# CampusOne AI Architecture

User
  |
  v
Chat Interface
  |
  v
FastAPI Backend
  |
  v
Intent / Domain Router
  |
  +----------------+----------------+----------------+
  |                |                |                |
  v                v                v                v
 IT              HR_ADMIN          FEES          FACILITIES
  |                |                |                |
  v                v                v                v
IT RAG          Admin RAG        Fees RAG       Facilities RAG
  |                |                |                |
  +----------------+----------------+----------------+
                           |
                           v
                    LLM / Answer Generator
                           |
                           v
                    Grounded Answer
                           |
                           v
                     Source Citation