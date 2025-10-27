from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import completion
import json
import os
from dotenv import load_dotenv
from collections import defaultdict, Counter
import re
import math

# --------------------------------------------------------------------------------------
# Env Path and file load 
# --------------------------------------------------------------------------------------

env_path = r"C:\\Users\\cyril\\OneDrive\\Desktop\\ITC\\Project3\\SAGE\\.env"
load_dotenv(dotenv_path=env_path)
print("Loaded .env from:", env_path)
print("OPENAI_API_KEY loaded?", bool(os.getenv("OPENAI_API_KEY")))

# --------------------------------------------------------------------------------------
# SENTIMENT ANALYSIS FUNCTION (minute-level, with "minute": "0 to 1" format)
# --------------------------------------------------------------------------------------

def analyze_sentiment_by_minute(conversation: list) -> dict:
    """
    Perform sentiment analysis for each message, then aggregate by minute range (0 to 1, 1 to 2, etc.).
    Each entry is: [start_time_sec, end_time_sec, speaker, text]
    """

    def safe_parse_json(raw):
        """Safely parse model output even if wrapped in markdown."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    message_results = []

    # ---- Run sentiment analysis for each message ---- #
    for entry in conversation:
        start_t, end_t, speaker, text = entry

        resp = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise sentiment analysis model for customer service calls. "
                        "Detect fine emotions such as anger, frustration, calm, apology, or satisfaction. "
                        "Return only JSON: {\"label\": <emotion>, \"score\": <0–1>}."
                    ),

                },
                {"role": "user", "content": text},
            ],
        )

        raw = resp["choices"][0]["message"]["content"]
        parsed = safe_parse_json(raw)

        if parsed:
            label = parsed.get("label", "neutral")
            score = float(parsed.get("score", 0.5))
        else:
            label, score = "neutral", 0.5

        message_results.append({
            "start_t": start_t,
            "end_t": end_t,
            "speaker": speaker,
            "label": label,
            "score": score
        })

    # ----------------------------------------------------------------------------------
    # GROUP BY MINUTE
    # ----------------------------------------------------------------------------------

    minute_buckets = defaultdict(list)

    for m in message_results:
        minute_index = int(math.floor(m["start_t"] / 60))
        minute_buckets[minute_index].append(m)

    minute_summary = []

    for minute, msgs in sorted(minute_buckets.items()):
        label_counts = Counter(msg["label"] for msg in msgs)
        label_scores = defaultdict(float)
        for msg in msgs:
            label_scores[msg["label"]] += msg["score"]

        avg_scores = {l: label_scores[l] / label_counts[l] for l in label_counts}
        top_label = max(label_counts, key=label_counts.get)
        top_score = round(avg_scores[top_label], 2)

        # Format minute as "0 to 1", "1 to 2", etc.
        minute_label = f"{minute} to {minute + 1}"

        minute_summary.append({
            "minute": minute_label,
            "label": top_label,
            "score": top_score,
            "message_count": len(msgs)
        })

    # ----------------------------------------------------------------------------------
    # OVERALL SENTIMENT
    # ----------------------------------------------------------------------------------

    all_labels = [m["label"] for m in message_results]
    all_scores = defaultdict(float)
    label_counts = Counter(all_labels)

    for m in message_results:
        all_scores[m["label"]] += m["score"]

    avg_scores = {l: all_scores[l] / label_counts[l] for l in label_counts}
    overall_label = max(label_counts, key=label_counts.get)
    overall_score = round(avg_scores[overall_label], 2)

    # ----------------------------------------------------------------------------------
    # RETURN STRUCTURED RESULT
    # ----------------------------------------------------------------------------------
    return {
        "sentiment_overall": overall_label,
        "overall_score": overall_score,
        "granularity": "1-minute",
        "timeline": minute_summary
    }

# --------------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    conversation = [
        [1.7760000000000002,2.3760000000000003,"A","Hello."],
        [3.1260000000000003,3.676,"B","Hi."],
        [3.976,5.976,"B","Thank you for calling JB Morgan Chase."],
        [6.076,7.026,"B","My name is Ahmed."],
        [7.226,8.1260000000000012,"B","How can I help you?"],
        [9.1760000000000019,11.076000000000002,"A","Well, how can you help me?"],
        [13.326000000000002,17.776,"B","Whatever the issue is, we could discuss and have it solved."],
        [18.226,20.076,"B","Can I know your first name, sir?"],
        [21.026,24.876,"A","My name is Jeff, and my credit card isn't working."],
        [26.484,28.434,"B","Oh, I'm so sorry to hear that Jeff."],
        [28.884,34.734,"B","All right, I'll just need a few basic details from you and we'll have a look into it."],
        [34.884,36.234,"B","So what's your full name?"],
        [37.334,39.084,"A","Jeff Jefferson,"],
        [39.384,40.434000000000005,"A","January"],
        [40.434000000000005,41.484,"B","All right, Mr."],
        [41.584,41.884,"B","Jefferson,"],
        [41.984000000000009,43.584,"B","can I know your birth date?"],
        [45.984000000000009,47.634,"A","1st, 1990."],
        [48.984000000000009,49.984000000000009,"B","Okay, thank you."],
        [50.134,53.084,"B","Also, can I, if you have your account number handy,"],
        [53.234,54.384,"B","can I get that?"],
        [56.148,56.748000000000005,"A","Fine."],
        [56.798,58.048,"A","Let me find it."],
        [59.098,62.298,"A","5-5-5-5-5-5-5-5-5."],
        [63.998000000000005,65.798,"B","Alright, thank you, Mr."],
        [65.948000000000008,66.348,"B","Jefferson."],
        [66.548,71.048,"B","So, I'm sorry I could not see any record in my system."],
        [73.848000000000013,75.39800000000001,"A","So much help you were."],
        [77.448000000000008,79.39800000000001,"B","So sorry, could you repeat,"],
        [79.64800000000001,82.14800000000001,"B","can I know your debit card number?"],
        [83.38,85.38,"A","Do you need my social security number too?"],
        [85.47999999999999,88.28,"B","No, that won't be necessary."],
        [88.679999999999993,92.22999999999999,"B","Can I just have your credit card number so I could look at you in my system?"],
        [93.72999999999999,94.38,"A","Let's see,"],
        [94.72999999999999,99.97999999999999,"A","uh, two one two three three four five nine nine two."],
        [100.72999999999999,101.33,"B","Okay,"],
        [101.43,103.43,"B","and can I know the CVV on it?"],
        [104.38,105.18,"A","Three twenty."],
        [109.14,116.09,"B","Alright, thank you Mr. Jefferson. Sorry I'm so sorry for your inconvenience. Um what were you calling regarding?"],
        [117.19,118.69,"A","That credit card isn't working."],
        [120.44,124.54,"B","Okay, uh what issues or challenges are you facing?"],
        [124.69,125.44,"B","Okay,"],
        [125.79,129.74,"A","When I try to go to pay for something it says something's wrong."],
        [131.84,136.24,"B","do you know any specific or when was the card activated?"],
        [137.492,139.59199999999998,"A","I don't know, like a few months ago?"],
        [140.642,144.042,"B","Well the card show is fine on our system okay"],
        [148.09199999999998,149.44199999999998,"A","Well, it's not fine."],
        [153.94199999999998,160.742,"B","let me transfer my call let me transfer a call to the technical team they might help you out please"],
        [160.742,161.242,"A","Fine."],
        [161.242,162.94199999999998,"B","hold on the line"]
        ]

    result = analyze_sentiment_by_minute(conversation)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    #with open("minute_sentiment_output.json", "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=2, ensure_ascii=False)
    #print("\n Sentiment analysis saved to minute_sentiment_output.json")
