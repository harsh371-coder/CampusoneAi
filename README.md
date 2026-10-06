# CampusOne AI

## One Front Door for Everything

CampusOne AI is a university assistant designed for Bennett University.

The system accepts a student's question through one chat interface,
understands the topic, routes the question to the correct knowledge
area, retrieves relevant university information, and generates a
grounded answer with sources.

## Supported Domains

- IT
- HR_ADMIN
- FEES
- FACILITIES

## Core Capabilities

- Topic/domain routing
- Intent detection
- Routing confidence
- Clarification when uncertain
- Multi-topic questions
- Grounded answers using university knowledge
- Source citation
- Routing accuracy evaluation
- End-to-end resolution evaluation

## Technology

- Python 3.11
- FastAPI
- Mistral
- Sentence Transformers
- FAISS
- PyMuPDF
- GitHub

## Project Structure

backend/
- router/
- rag/
- agents/
- schemas/
- config/
- services/

data/
- raw/
- processed/

evaluation/
tests/
docs/

## Development

Each team member uses their own Python virtual environment.

The virtual environment is NOT committed to Git.

All members use the same requirements.txt.

## Branches

main   → stable project
develop → integration branch
feature/* → individual work