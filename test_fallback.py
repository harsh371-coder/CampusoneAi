from backend.router.fallback import keyword_route


questions = [
    "My university WiFi is not working.",
    "My university email is not working.",
    "How do I reset my university password?",
    "When are my semester fees due?",
    "How can I pay my university fees?",
    "How do I apply for revaluation?",
    "What bus routes are available?",
    "What are the hostel facilities?",
    "The AC in my room is broken.",
    "My WiFi is not working and I also want to know my fee deadline.",
    "My payment isn't working.",
    "Write a C++ program for quicksort.",
]


for question in questions:

    result = keyword_route(question)

    print("\n" + "=" * 80)

    print("QUESTION:")
    print(question)

    print("\nRESULT:")
    print(result.model_dump_json(indent=2)) 