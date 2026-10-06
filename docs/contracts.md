# CampusOne AI - Shared Contracts

This document defines how the different backend components
communicate with each other.

---

## 1. Router

### Input

A student's natural-language question.

### Output

The router must provide:

- domains
- primary_domain
- intent
- confidence
- needs_clarification
- clarification_question
- is_out_of_scope
- routing_method

---

## 2. Retriever

### Input

- user question
- selected domain

### Output

Relevant knowledge chunks with:

- text
- source title
- source URL
- page/section where available
- relevance score
- domain

---

## 3. Agent / Answer Generator

### Input

- user question
- retrieved knowledge
- conversation context

### Output

- final answer
- sources
- answer confidence where supported

---

## 4. API

The API is responsible for connecting:

User
→ Router
→ Retriever
→ Agent
→ Final Response

---

## Supported Domains

IT
HR_ADMIN
FEES
FACILITIES