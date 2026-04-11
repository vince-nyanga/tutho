import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.local_client import LocalClient
from src.router import Router
from src.tools.curriculum import CurriculumStore

async def main():
    client = LocalClient()
    curriculum = CurriculumStore()
    router = Router(client, curriculum)

    session = {"grade": 12, "subject": "Mathematics"}

    test_messages = [
        "Help me with sequences",
        "What is the difference between arithmetic and geometric sequences?",
        "How do I find the nth term?",
    ]

    for msg in test_messages:
        print(f"\n{'=' * 60}")
        print(f"Student: {msg}")
        print(f"{'=' * 60}")
        result = await router.handle_message(msg, session)
        print(f"\nThuto AI: {result}")


asyncio.run(main())
