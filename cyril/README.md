

1) Features
    *)Minute-by-minute sentiment timeline (0 to 1, 1 to 2, etc.)
    *)Calculates the dominant emotion and confidence score for each minute
    *)Overall sentiment summary
    *)Output JSON

2) Workflow

    *) Load Environment – The script loads your .env file to access the OPENAI_API_KEY.
    *) Input Conversation – A list of messages is provided, each with a start time, end time, speaker, and text.
    *) Run Sentiment Analysis–Each message is sent to the GPT model (gpt-4o) using litellm.completion() for emotion detection.
    *) Aggregate by Minute – Emotions are grouped into 1-minute intervals (0 to 1, 1 to 2, etc.).
    *) Compute Scores – The script calculates the dominant emotion and average confidence score for each minute.
    *) Summarize Results – It generates an overall sentiment label and score for the entire conversation.
    *) Output JSON – Results are printed or optionally saved to minute_sentiment_output.json.


3) Main Function — analyze_sentiment_by_minute()

    Takes a conversation list (each line with start time, end time, speaker, and message)
    and returns a minute-by-minute sentiment summary.

4) Sub-function: safe_parse_json()
    When GPT returns its output, it might be wrapped in triple backticks (```json ... ```).
    This function safely removes that formatting and tries to parse valid JSON.
    If parsing fails, it returns None instead of crashing.

5) Running Sentiment Analysis for Each Message
    for entry in conversation:
        start_t, end_t, speaker, text = entry
    
    *) Loops through every line of the conversation.

        Each entry looks like:
        [start_time_sec, end_time_sec, speaker, text]
        Then it sends the text to GPT-4o:

    *) The system message defines the role and instructs GPT to return emotion + score as JSON.
        The user message is the actual text to analyze (the speaker’s dialogue).
    
    *) Each analyzed message is stored as:
             {
                "start_t": ...,
                "end_t": ...,
                "speaker": ...,
                "label": ...,
                "score": ...
             }
6️)  Grouping by Minute
        minute_buckets = defaultdict(list)
        for m in message_results:
            minute_index = int(math.floor(m["start_t"] / 60))
            minute_buckets[minute_index].append(m)
        *) Groups all messages that fall within the same minute range (e.g., 0–1, 1–2, etc.).
        *) minute_index = start_time divided by 60 (then rounded down).

7) Calculating Dominant Emotion per Minute 
    for minute, msgs in sorted(minute_buckets.items()):
        label_counts = Counter(msg["label"] for msg in msgs)
        label_scores = defaultdict(float)

    *) Counts how many times each emotion appears in that minute.
    *) Averages the scores for each label.
    *) Selects the most frequent emotion (top_label) as the dominant mood of that minute.

8️)  Calculating Overall Conversation Sentiment

    all_labels = [m["label"] for m in message_results]
    ...
    overall_label = max(label_counts, key=label_counts.get)
    overall_score = round(avg_scores[overall_label], 2)

    *) Aggregates all emotions across the conversation.
    *) Finds the most common emotion overall (e.g., “frustration”).
    *) Computes the average confidence score for that emotion.