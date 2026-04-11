import asyncio
from src.local_client import LocalClient
from src.tools.curriculum import CurriculumStore
from src.router import Router

async def main():
    client = LocalClient()
    curriculum = CurriculumStore()
    router = Router(client, curriculum)

    test_messages = [
        "Help me with sequences",
        "Give me a question on calculus",
        "What tips for paper 1?",
        "What should I study next?",
        "Hi",
        "Who won the rugby?",
        "Ngicela ungichazele nge-arithmetic sequence",
    ]

    session = {}
    for msg in test_messages:
        result = await router._classify(msg, session)
        print(f"\n'{msg}'")
        print(f"  → {result}")


asyncio.run(main())