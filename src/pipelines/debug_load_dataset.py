from src.data.loaders import load_questions_from_config


def main():
    # این مسیر / کانفیگ را بسته‌به چیزی که می‌خواهی تست کنی تنظیم کن
    config = {
        "version": "gpqa_main",
        "subject_filter": None,       # یا "Physics" / "Biology" / "Chemistry"
        "subject_mode": "high_level", # یا "subdomain"
        "difficulty_filter": None,
    }

    questions = load_questions_from_config("gpqa", config)

    print(f"[debug] Loaded {len(questions)} GPQA questions.")
    if questions:
        q = questions[0]
        print("[debug] Sample question metadata (no text shown):")
        print("  id:", q.id)
        print("  subject:", q.subject)
        print("  difficulty:", q.difficulty)
        print("  num_options:", len(q.options) if q.options else 0)
        # متن سوال را چاپ نمی‌کنیم تا با شروط GPQA تضاد نداشته باشد.


if __name__ == "__main__":
    main()
