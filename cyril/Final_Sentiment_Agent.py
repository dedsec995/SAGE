import os
import re
import json
import math
from dotenv import load_dotenv
from collections import defaultdict, Counter
from litellm import completion

# --------------------------------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------------------------------
env_path = r"C:\\Users\\cyril\\OneDrive\\Desktop\\ITC\\Project3\\SAGE\\.env"
load_dotenv(dotenv_path=env_path)
print("Loaded .env from:", env_path)
print("OPENAI_API_KEY loaded?", bool(os.getenv("OPENAI_API_KEY")))

# --------------------------------------------------------------------------------------
# Safe JSON parser
# --------------------------------------------------------------------------------------
def safe_parse_json(raw):
    """Safely parse model output even if wrapped in markdown."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

# --------------------------------------------------------------------------------------
# Main sentiment analysis function
# --------------------------------------------------------------------------------------
def analyze_sentiment_by_minute(conversation: list) -> dict:
    """
    Group messages by minute, send combined text to GPT, and return sentiment summary.
    Each entry in conversation: [start_time_sec, end_time_sec, speaker, text]
    """
    # ---- Group messages by minute ----
    minute_buckets = defaultdict(list)
    for entry in conversation:
        start_t, end_t, speaker, text = entry
        minute_index = int(math.floor(start_t / 60))
        minute_buckets[minute_index].append((speaker, text))

        #print(minute_buckets)

    # ---- Run sentiment analysis per minute ----
    minute_summary = []
    for minute, msgs in sorted(minute_buckets.items()):
        combined_text = " ".join([f"{speaker}: {text}" for speaker, text in msgs])

        # Print what’s being analyzed
        #print("\n" + "="*80)
        #print(f" Minute Range: {minute} to {minute + 1}")
        #print(f"  Combined Conversation Text:\n{combined_text}")

        resp = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise emotion detection model for customer conversations. "
                        "Analyze the following 1-minute transcript and identify the *dominant emotion* clearly. "
                        "Differentiate carefully between: "
                        "Anger (aggressive, raised voice), "
                        "Frustration (annoyed or impatient tone), "
                        "Calm (neutral or polite tone), "
                        "Apology (expressing regret), and "
                        "Satisfaction (happy or thankful tone). "
                        "Return only JSON: {\"label\": <emotion>, \"score\": <0–1>}."
                    ),
                },
                {"role": "user", "content": combined_text},
            ],
        )

        # Raw GPT output
        raw = resp["choices"][0]["message"]["content"]
        #print("\n Raw GPT Output:")
        #print(raw)

        # Parse safely
        parsed = safe_parse_json(raw)
        if parsed:
            label = parsed.get("label", "neutral")
            score = float(parsed.get("score", 0.5))
        else:
            label, score = "neutral", 0.5

        # Show parsed result
        #print(f" Parsed Emotion: {label} | Score: {score}")

        # Add to summary
        minute_label = f"{minute} to {minute + 1}"
        minute_summary.append({
            "minute": minute_label,
            "label": label,
            "score": round(score, 2),
            "message_count": len(msgs)
        })

    # ---- Compute overall sentiment ----
    label_counts = Counter(m["label"] for m in minute_summary)
    #print(label_counts)
    score_totals = defaultdict(float)
    #print(score_totals)
    for m in minute_summary:
        score_totals[m["label"]] += m["score"]

    avg_scores = {l: score_totals[l] / label_counts[l] for l in label_counts}
    #print(avg_scores)
    overall_label = max(label_counts, key=label_counts.get)
    #print(overall_label)
    overall_score = round(avg_scores[overall_label], 2)
    #print(overall_score)
    # ---- Return structured result ----
    return {
        "sentiment_overall": overall_label,
        "overall_score": overall_score,
        "granularity": "1-minute",
        "timeline": minute_summary
    }

# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    conversation = [
    # 0–1 min → Anger
    [5.0, 10.0, "A", "This is unacceptable! I’ve been waiting for 20 minutes and nobody’s helping me!"],
    [15.0, 20.0, "B", "Sir, please lower your voice. I’m here to help, but I need to know what happened."],
    [30.0, 50.0, "A", "What happened? Your service failed! My card got blocked for no reason!"],

    # 1–2 min → Frustration
    [65.0, 75.0, "A", "I already told you my account number twice. Why do you keep asking again?"],
    [80.0, 90.0, "B", "I’m just verifying your details for security purposes, sir."],
    [95.0, 100.0, "A", "This is so slow. You people really need to fix your systems."],

    # 2–3 min → Calm
    [125.0, 135.0, "B", "Thank you for confirming, Mr. Davis. I’m checking your account right now."],
    [140.0, 150.0, "A", "Okay, I understand. Please take your time."],
    [155.0, 165.0, "B", "Everything seems fine on our end. Let’s try a small reset, shall we?"],

    # 3–4 min → Apology
    [185.0, 190.0, "A", "I’m sorry for being rude earlier. I was just really stressed out."],
    [195.0, 205.0, "B", "That’s completely fine, I understand your frustration. Thank you for apologizing."],
    [210.0, 220.0, "A", "I appreciate your patience too, really."],

    # 4–5 min → Satisfaction
    [245.0, 255.0, "B", "Looks like the issue is fixed now, your card should work again."],
    [260.0, 270.0, "A", "Oh, thank you so much! It finally went through."],
    [275.0, 285.0, "A", "You’ve been really helpful today, I appreciate it!"]
]


    result = analyze_sentiment_by_minute(conversation)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("= # ="*20)

    #with open("minute_sentiment_output.json", "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=2, ensure_ascii=False)
    #print("\n Sentiment analysis saved to minute_sentiment_output.json")

    c=conversation = [
    # 0 to 1 min → Angry tone
    [10.0, 20.0, "A", "I have been waiting for so long! This is ridiculous!"],
    [40.0, 50.0, "B", "Sir, please calm down, we are trying our best."],

    # 1.1 to 1.59 min → Frustrated tone
    [70.0, 80.0, "A", "I already gave you my details twice. Why do you keep asking?"],
    [90.0, 95.0, "B", "I understand, but I need to verify them again for security."],

    # 2 to 3 min → Calm tone
    [125.0, 135.0, "A", "thank you im sorry ."],
    [150.0, 170.0, "B", "its okay , have a good day ."]
]

    print("==="*50)

    sentiment_result = analyze_sentiment_by_minute(c)
    print(json.dumps(sentiment_result, indent=2, ensure_ascii=False))
