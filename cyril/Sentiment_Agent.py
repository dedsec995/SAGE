from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import completion
import json
import os
from dotenv import load_dotenv
from collections import defaultdict, Counter
import re

# --------------------------------------------------------------------------------------
# Env Path and file load 
# --------------------------------------------------------------------------------------

env_path = r"C:\\Users\\cyril\\OneDrive\\Desktop\\ITC\\Project3\\SAGE\\.env"
load_dotenv(dotenv_path=env_path)
print("Loaded .env from:", env_path)
print("OPENAI_API_KEY loaded?", bool(os.getenv("OPENAI_API_KEY")))

# --------------------------------------------------------------------------------------
# SENTIMENT ANALYSIS FUNCTION (with speaker-level sentiment)
# --------------------------------------------------------------------------------------

def analyze_sentiment(conversation: list) -> dict:
    """
    Perform sentiment analysis for each message, summarize per speaker and overall.
    Each entry: [start_time, end_time, speaker, text]
    """
    def safe_parse_json(raw):
        """Safely parse model output even if wrapped in markdown."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    timeline = []

    # ---- Run sentiment analysis for each message ---- #
    for entry in conversation:
        start_t, end_t, speaker, text = entry

        resp = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise sentiment analysis model. "
                        "Return a JSON object: {\"label\": <emotion>, \"score\": <0–1>} "
                        "Respond ONLY in JSON without markdown or code blocks."
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

        timeline.append({
            "t": start_t,
            "speaker": speaker,
            "label": label,
            "score": score
        })

    # ----------------------------------------------------------------------------------
    # SPEAKER-WISE SENTIMENT CALCULATION
    # ----------------------------------------------------------------------------------

    speaker_data = defaultdict(list)
    for t in timeline:
        speaker_data[t["speaker"]].append(t)

    speaker_summary = {}
    for speaker, messages in speaker_data.items():
        label_counts = Counter(m["label"] for m in messages)
        label_scores = defaultdict(float)
        for m in messages:
            label_scores[m["label"]] += m["score"]

        avg_scores = {label: label_scores[label] / label_counts[label] for label in label_counts}
        top_label = max(label_counts, key=label_counts.get)
        top_score = round(avg_scores[top_label], 2)

        speaker_summary[speaker] = {
            "sentiment_overall": top_label,
            "overall_score": top_score
        }

    # ----------------------------------------------------------------------------------
    # OVERALL SENTIMENT (entire conversation)
    # ----------------------------------------------------------------------------------

    label_counts = Counter(t["label"] for t in timeline)
    label_scores = defaultdict(float)
    for t in timeline:
        label_scores[t["label"]] += t["score"]

    avg_scores = {label: label_scores[label] / label_counts[label] for label in label_counts}
    overall_label = max(label_counts, key=label_counts.get)
    overall_score = round(avg_scores[overall_label], 2)

    # ----------------------------------------------------------------------------------
    # RETURN STRUCTURED RESULT
    # ----------------------------------------------------------------------------------
    return {
        "sentiment_overall": overall_label,
        "overall_score": overall_score,
        "granularity": "1m",
        "start_time": conversation[0][0],
        "speakers": speaker_summary,
        "timeline": timeline,
    }

# --------------------------------------------------------------------------------------
# AGENT WRAPPER 
# --------------------------------------------------------------------------------------

sentiment_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-4o"),
    name="sentiment_agent",
    description="Analyzes emotional tone across conversation turns.",
    instruction=(
        "Perform fine-grained sentiment analysis on text segments. "
        "Return structured JSON with overall sentiment and a timeline."
    ),
    tools=[analyze_sentiment],
)

# --------------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    conversation = [
        [0.0, 1.5, "A", "Hi, I’ve been on hold for thirty minutes!"],
        [2.0, 3.2, "B", "I’m really sorry about that, sir. How can I help you today?"],
        [3.6, 6.8, "A", "You can start by actually fixing my internet that’s been down for TWO days!"],
        [7.2, 9.2, "B", "I understand your frustration. Let me check your account details."],
        [10.0, 12.0, "A", "I’ve already given my details three times today!"],
        [13.0, 15.0, "B", "Could you please confirm your account number once more?"],
        [15.5, 17.5, "A", "This is ridiculous. It’s 2259-88-3145."],
        [18.0, 20.0, "B", "Thank you. Give me just a moment to pull it up."],
        [22.0, 25.0, "A", "Yeah, sure, take your time like everyone else apparently."],
        [26.0, 28.0, "B", "Okay, I can see your connection was suspended due to a system error."],
        [28.3, 30.3, "A", "A system error? So I suffer because of YOUR system?"],
        [31.0, 34.0, "B", "I completely understand, sir. I’ll do my best to restore it right away."],
        [34.5, 37.0, "A", "I pay on time every month. This is unacceptable."],
        [37.5, 39.0, "B", "You’re absolutely right, and I apologize sincerely."],
        [40.0, 42.0, "B", "I’ve just sent a reset command to your router."],
        [43.0, 44.5, "A", "Nothing’s changing here!"],
        [45.0, 47.0, "B", "It may take about a minute to fully reconnect."],
        [48.0, 50.0, "A", "Unbelievable. Always excuses."],
        [51.0, 54.0, "B", "I really am trying to help, sir. Could you restart your modem for me?"],
        [55.0, 57.0, "A", "Fine, but if this doesn’t work, I’m canceling everything."],
        [57.5, 60.0, "B", "Understood. Please let me know once it’s restarted."],
        [62.0, 64.0, "A", "Okay, it’s restarting."],
        [66.0, 68.0, "B", "Alright, I’m seeing some signal now."],
        [69.0, 71.0, "A", "About time."],
        [72.0, 74.0, "B", "Please check if the connection light is solid green."],
        [75.0, 76.0, "A", "Yeah... looks green."],
        [77.0, 79.0, "B", "Perfect! Try loading a page now."],
        [80.0, 82.0, "A", "Okay… wait— it’s working."],
        [83.0, 85.0, "B", "That’s great news!"],
        [85.5, 87.0, "A", "Finally! Took forever though."],
        [88.0, 90.0, "B", "I completely understand, and I’ve added a one-month credit for the trouble."],
        [91.0, 93.0, "A", "Good. At least that’s something."],
        [94.0, 95.5, "B", "Is there anything else I can assist you with today?"],
        [96.0, 98.0, "A", "No, just make sure this doesn’t happen again."],
        [99.0, 101.0, "B", "We’ll certainly do our best, sir."],
        [102.0, 104.0, "A", "Yeah, you’d better."],
        [105.0, 107.0, "B", "Thank you for calling TechLine Support, have a better day ahead."],
        [108.0, 109.5, "A", "We’ll see about that."]
    ]

    result = analyze_sentiment(conversation)

    # Print structured JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save to file
    with open("sentiment_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n✅ Sentiment analysis saved to sentiment_output.json")
