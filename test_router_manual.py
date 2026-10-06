from backend.router.router import IntentRouter


router = IntentRouter()


questions = [
    "My university WiFi is not working.",
    "My university email is not working.",
    "How do I reset my university password?",
    "When are my semester fees due?",
    "How can I pay my university fees?",
    "How do I apply for revaluation?",
    "What bus routes are available?",
    "What are the hostel facilities?",
    "My WiFi is not working and I also want to know my fee deadline.",
    "My payment isn't working.",
    "Write a C++ program for quicksort.",
]


for question in questions:

    print("\n" + "=" * 80)

    print("QUESTION:")
    print(question)

    result = router.route(question)

    print("\nRESULT:")
    print(result.model_dump_json(indent=2))