# Authors: Nicholas Lombard(26210827) and Ben Morton()
import os
from groq import Groq
from transformers import pipeline
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_response(Prompt):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        client = Groq(api_key=api_key) 
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": Prompt,
                }
            ],
            model="llama-3.2-11b-text-preview",
        )
        return str(chat_completion.choices[0].message.content)
    
    except Exception as e:
        return "FAILED"
    
    
def batch_summarize(reviews, max_length=200, min_length=50):
    """
    Summarize each batch of reviews.

    Args:
        reviews (list of str): List of reviews to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        list: A list of summarized reviews.
    """
    summaries = []
    for review in reviews:
        combined_review = " ".join(review)

        if len(combined_review.split()) <= min_length:
            summaries.append(combined_review)
        else:
            summary = summarizer(combined_review, max_length=max_length, min_length=30, truncation=True)
            summaries.append(summary[0]['summary_text'])
    return summaries


def funnel_batch_summarization(reviews, initial_batch_size=20, target_length=1000):
    """
    Perform iterative funnel-shaped batch summarization until the target length is reached.
    
    Args:
        reviews (list of str): List of reviews to summarize.
        initial_batch_size (int): Initial number of reviews per batch.
        target_length (int): Target length (in words) for the final summary.
        
    Returns:
        str: Final summarized text.
    """
    batch_size = initial_batch_size
    combined_reviews = [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]  # Batching the reviews
    
    while True:
        summarized_batches = batch_summarize(combined_reviews, max_length=250, min_length=50)
        print("length of list: " + str(len(summarized_batches)))
        batch_size = 4
        print(summarized_batches[0])
        combined_reviews = [summarized_batches[i:i + batch_size] for i in range(0, len(summarized_batches), batch_size)]
        combined_summary = " ".join([batch for batch in summarized_batches])
        if len(combined_summary.split()) <= target_length:
            return combined_summary



def generate_positive(content):
    prompt = "You are a part of a professional Brief-generating agent that creates concise, polished paragraphs summarizing user feedback. Your task is to generate a well-written paragraph based on the provided summary of positive reviews. Your goal is to highlight the key aspects that users appreciate most about the product or service, focusing on specific features, benefits, or experiences. Write in a professional, engaging tone that reflects customer satisfaction and emphasizes strengths. Do not include any introduction, summary, or additional headings—only produce the paragraph itself. Ensure the writing flows smoothly and reads naturally, without sounding mechanical or repetitive. Below is the summary of the reviews for this topic: \n"
    return generate_response((prompt + str(content)))

def generate_negative(content):
    prompt = "You are a part of a professional Brief-generating agent that creates concise, polished paragraphs summarizing user feedback. Your task is to generate a well-written paragraph based on the provided summary of negative reviews. Your goal is to highlight key areas where users are dissatisfied, focusing on challenges, frustrations, bugs or areas of improvement. Write in a professional and objective tone, suggesting these insights as opportunities for improvement without being overly negative. Do not include any introduction, summary, or additional headings—only produce the paragraph itself. Ensure the writing flows smoothly and reads naturally, without sounding mechanical or repetitive.Below is the summary of the reviews for this topic: \n"
    return generate_response((prompt + str(content)))
    
def generate_heading(content):
    prompt = "You are a part of a professional Brief-generating agent. Your task is to create a **creative, engaging, and concise heading** that captures the essence of the topic discussed in the provided paragraph. The heading should reflect the **main theme or message** conveyed in the paragraph, while being **professional, insightful, and appropriate** for use in a formal report or presentation. Aim for a **compelling and concise title (max 8 words)** that draws attention and encapsulates the key idea discussed in the paragraph without exaggerating or oversimplifying it.Only return the heading, with no additional content.Below is the paragraph:\n"
    return generate_response((prompt + str(content)))

def generate_pos_improve(content):
    prompt = "You are part of a professional Brief-generating agent. Your task is to generate a **polished, thoughtful paragraph** that provides **insights on how the company could leverage the positive aspects** discussed in the provided content to further improve their product or service. Your response should only be a **single paragraph**. Do not include any introduction, summary, or extra content—only the paragraph itself.If there are **meaningful and relevant insights**, include them in a constructive tone, focusing on **opportunities for growth or ways to enhance the user experience**. Avoid assumptions or forced suggestions if the content does not provide clear insights. If no meaningful insights are found, respond with a **balanced statement** that highlights the overall strengths.Write in a **professional, non-pushy tone** that flows smoothly and avoids repetition. Below is the content you need to work from: \n"
    return generate_response((prompt + str(content)))

def generate_neg_improve(content):
    prompt = "You are part of a professional Brief-generating agent. Your task is to generate a **concise, constructive paragraph** focusing on the **most serious bugs, deficiencies, or weaknesses** identified in the provided content. Your response should only be a **single paragraph**. Do not include any introduction, summary, or extra content—only the paragraph itself. If no meaningful issues or actionable insights are identified, respond with a neutral and balanced statement about the general feedback without forcing recommendations. Use a **professional, objective tone** that offers constructive feedback, prioritizing serious issues where fixing them would improve the product or service.Below is the content you need to work from: \n"
    return generate_response((prompt + str(content)))

def generate_conclusion(content):
    prompt = "You are part of a professional Brief-generating agent. Your task is to generate a **concise, polished conclusion** that reflects the overall themes discussed in the brief. The conclusion should refer back to key points or patterns identified throughout the brief, reinforcing the central ideas in a professional and succinct way. Do not introduce new information, provide a summary, or add any extra content—only generate the conclusion itself. Ensure the writing flows naturally and leaves a cohesive final impression.Below is the brief you need to work from: \n"
    return generate_response((prompt + str(content)))

def generate_intro(content):
    prompt = "You are part of a professional Brief-generating agent. Your task is to generate a **concise introduction** that briefly outlines the purpose of the report and what it aims to achieve. The introduction should set the context for the report by explaining that it summarizes key user feedback, highlighting both positive aspects and areas for improvement. Do not provide a summary of the contents or any extra information—only generate the introduction itself. Keep the tone professional and succinct. Below is the brief you need to introduce: \n"
    return generate_response((prompt + str(content)))



folder_path = 'Data'
positive_brief = []
negative_brief = []
headings_pos = []
headings_neg = []

for file_name in os.listdir(folder_path):
    print(file_name)
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        reviews = df['content'].tolist()
        summary = funnel_batch_summarization(reviews, initial_batch_size=10, target_length=1000)
        if "negative" in file_name:
            negative_brief.append(str(generate_negative(summary)))
            headings_neg.append(str(generate_heading(negative_brief[-1])))
        else:
            positive_brief.append(str(generate_positive(summary)))
            headings_pos.append(str(generate_heading(negative_brief[-1])))
            

actions_neg = generate_neg_improve((" ".join(negative_brief)))
actions_pos = generate_neg_improve((" ".join(positive_brief)))
actions_neg_heading = "Areas to Improve:"
actions_pos_heading = "Positives to Leverage:"

formatted_report = []

if headings_pos:
    for heading, brief in zip(headings_pos, positive_brief):
        formatted_report.append(f"{heading}\n{brief}\n")

formatted_report.append(f"{actions_pos_heading}\n{actions_pos}\n")

if headings_neg:
    for heading, brief in zip(headings_neg, negative_brief):
        formatted_report.append(f"{heading}\n{brief}\n")

formatted_report.append(f"{actions_neg_heading}\n{actions_neg}\n")

final_report = "\n".join(formatted_report)

output_file_path = 'final_report_Approach1.txt'

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(final_report)

print(f"Report written to {output_file_path}")