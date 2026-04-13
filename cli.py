#!/usr/bin/env python3
"""
CLI interface for DocMind RAG engine.

Examples:
  python cli.py ingest --path ./docs/paper.pdf
  python cli.py ingest --dir ./docs/
  python cli.py query "What is the main finding?"
  python cli.py chat   # interactive loop
"""

import argparse
import os
import sys
from pathlib import Path


def get_engine():
    from app.rag_engine import RAGEngine
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not anthropic_key:
        print("Error: Set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    if not openai_key:
        print("Error: Set the OPENAI_API_KEY environment variable (used for embeddings).")
        sys.exit(1)
    return RAGEngine(
        vector_store=os.getenv("VECTOR_STORE", "chroma"),
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
    )


def cmd_ingest(args):
    engine = get_engine()
    if args.path:
        result = engine.ingest_pdf(args.path)
        if result["status"] == "success":
            print(f"✓ Ingested {result['file']}: {result['pages']} pages, {result['chunks']} chunks")
        else:
            print(f"↩ Skipped {result['file']}: {result['reason']}")
    elif args.dir:
        results = engine.ingest_directory(args.dir)
        for r in results:
            print(f"  {'✓' if r['status']=='success' else '↩'} {r['file']}")


def cmd_query(args):
    engine = get_engine()
    result = engine.query(args.question)
    print(f"\n📝 Answer:\n{result['answer']}\n")
    if result["sources"]:
        print("📄 Sources:")
        for s in result["sources"]:
            print(f"  • {s['file']} (p.{s['page']}): {s['snippet'][:80]}...")


def cmd_chat(args):
    engine = get_engine()
    print("\n🤖 DocMind Chat  (type 'quit' to exit, 'clear' to reset memory)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "clear":
            engine.clear_memory()
            print("Memory cleared.\n")
            continue
        result = engine.query(question)
        print(f"\nDocMind: {result['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="DocMind RAG CLI")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest PDF(s)")
    p_ingest.add_argument("--path", help="Single PDF path")
    p_ingest.add_argument("--dir", help="Directory of PDFs")

    p_query = sub.add_parser("query", help="One-shot question")
    p_query.add_argument("question", help="Question to ask")

    sub.add_parser("chat", help="Interactive chat loop")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
