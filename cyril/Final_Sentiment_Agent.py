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
        print(f" Parsed Emotion: {label} | Score: {score}")

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
        [1.776, 2.376, "A", "Hello."],
        [3.126, 3.676, "B", "Hi."],
        [3.976, 5.976, "B", "Thank you for calling JB Morgan Chase."],
        [6.076, 7.026, "B", "My name is Ahmed."],
        [7.226, 8.126, "B", "How can I help you?"],
        [9.176, 11.076, "A", "Well, how can you help me?"],
        [13.326, 17.776, "B", "Whatever the issue is, we could discuss and have it solved."],
        [18.226, 20.076, "B", "Can I know your first name, sir?"],
        [21.026, 24.876, "A", "My name is Jeff, and my credit card isn't working."],
        [26.484, 28.434, "B", "Oh, I'm so sorry to hear that Jeff."],
        [28.884, 34.734, "B", "All right, I'll just need a few basic details from you and we'll have a look into it."],
        [34.884, 36.234, "B", "So what's your full name?"],
        [37.334, 39.084, "A", "Jeff Jefferson,"],
        [39.384, 40.434, "A", "January"],
        [40.434, 41.484, "B", "All right, Mr."],
        [41.584, 41.884, "B", "Jefferson,"],
        [41.984, 43.584, "B", "can I know your birth date?"],
        [45.984, 47.634, "A", "1st, 1990."],
        [48.984, 49.984, "B", "Okay, thank you."],
        [50.134, 53.084, "B", "Also, can I, if you have your account number handy,"],
        [53.234, 54.384, "B", "can I get that?"],
        [56.148, 56.748, "A", "Fine."],
        [56.798, 58.048, "A", "Let me find it."],
        [59.098, 62.298, "A", "5-5-5-5-5-5-5-5-5."],
        [63.998, 65.798, "B", "Alright, thank you, Mr."],
        [65.948, 66.348, "B", "Jefferson."],
        [66.548, 71.048, "B", "So, I'm sorry I could not see any record in my system."],
        [73.848, 75.398, "A", "So much help you were."],
        [77.448, 79.398, "B", "So sorry, could you repeat,"],
        [79.648, 82.148, "B", "can I know your debit card number?"],
        [83.38, 85.38, "A", "Do you need my social security number too?"],
        [85.48, 88.28, "B", "No, that won't be necessary."],
        [88.68, 92.23, "B", "Can I just have your credit card number so I could look at you in my system?"],
        [93.73, 94.38, "A", "Let's see,"],
        [94.73, 99.98, "A", "uh, two one two three three four five nine nine two."],
        [100.73, 101.33, "B", "Okay,"],
        [101.43, 103.43, "B", "and can I know the CVV on it?"],
        [104.38, 105.18, "A", "Three twenty."],
        [109.14, 116.09, "B", "Alright, thank you Mr. Jefferson. Sorry I'm so sorry for your inconvenience. Um what were you calling regarding?"],
        [117.19, 118.69, "A", "That credit card isn't working."],
        [120.44, 124.54, "B", "Okay, uh what issues or challenges are you facing?"],
        [124.69, 125.44, "B", "Okay,"],
        [125.79, 129.74, "A", "When I try to go to pay for something it says something's wrong."],
        [131.84, 136.24, "B", "do you know any specific or when was the card activated?"],
        [137.49, 139.59, "A", "I don't know, like a few months ago?"],
        [140.64, 144.04, "B", "Well the card show is fine on our system okay"],
        [148.09, 149.44, "A", "Well, it's not fine."],
        [153.94, 160.74, "B", "let me transfer my call let me transfer a call to the technical team they might help you out please"],
        [160.74, 161.24, "A", "Fine."],
        [161.24, 162.94, "B", "hold on the line"]
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
