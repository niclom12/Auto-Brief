# Authors: Nicholas Lombard(26210827) and Ben Morton()
import os
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel, pipeline
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def tokenize_batch(reviews, max_length=512):
    reviews = [review for review in reviews if review.strip()]
    return tokenizer(reviews, padding=True, truncation=True, max_length=max_length, return_tensors="tf")


def generate_embeddings(reviews, batch_size=3):
    if len(reviews) == 0:
        raise ValueError("No reviews provided")
    embeddings = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_embeddings = sentence_model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return np.vstack([e.cpu().numpy() for e in embeddings])

def cluster_reviews(embeddings, num_clusters=5):
    """Cluster review embeddings using k-means."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans.cluster_centers_

def select_representative_reviews(reviews, embeddings, clusters, cluster_centers, top_n=2):
    """Select top representative reviews from each cluster."""
    selected_reviews = []
    for cluster_id in np.unique(clusters):
        
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_reviews = [reviews[i] for i in cluster_indices]
        cluster_embeddings = embeddings[cluster_indices]

        centroid = cluster_centers[cluster_id].reshape(1, -1)
        similarities = util.pytorch_cos_sim(cluster_embeddings, centroid).flatten()

        num_to_select = min(top_n, len(cluster_reviews))

        top_indices = np.argsort(similarities.numpy())[-num_to_select:]
        selected_reviews.extend([cluster_reviews[i] for i in top_indices])

    return selected_reviews

def generate_summary(content, min_length=10, max_length=700):
    """Generate a summary from the content using the summarization model."""
    if len(content.split()) < min_length:
        return content 

    summary = summarizer(content, max_length=min(max_length, 512), min_length=min_length, truncation=True)
    return summary[0]['summary_text']

def generate_response(Prompt):
    """Generate a response from Groq's LLM."""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": Prompt}],
            model="llama-3.2-11b-text-preview",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return "FAILED"

def process_reviews(reviews, num_clusters=12, top_n=2, min_length=50, max_length=150):
    """Main function to process reviews using clustering and summarization."""
    embeddings = generate_embeddings(reviews)

    clusters, cluster_centers = cluster_reviews(embeddings, num_clusters=num_clusters)

    selected_reviews = select_representative_reviews(reviews, embeddings, clusters, cluster_centers, top_n=top_n)

    combined_content = " ".join(selected_reviews)
    
    return generate_summary(combined_content, min_length=min_length, max_length=max_length)


def generate_positive(content):
    prompt = "You are a part of a professional Brief-generating agent that creates concise, polished paragraphs summarizing user feedback.\n Your task is to generate a well-written paragraph based on the provided summary of positive reviews. \nYour goal is to highlight the key aspects that users appreciate most about the product or service, focusing on specific features, benefits, or experiences. \nWrite in a professional, engaging tone that reflects customer satisfaction and emphasizes strengths. \nDo not include any introduction, summary, or additional headings—only produce the paragraph itself. Ensure the writing flows smoothly and reads naturally, without sounding mechanical or repetitive. \nBelow is the summary of the reviews for this topic: \n"
    return generate_response((prompt + str(content)))

def generate_negative(content):
    prompt = "You are a part of a professional Brief-generating agent that creates concise, polished paragraphs summarizing user feedback. Your task is to generate a well-written paragraph based on the provided summary of negative reviews. Your goal is to highlight key areas where users are dissatisfied, focusing on challenges, frustrations, bugs or areas of improvement. Write in a professional and objective tone, suggesting these insights as opportunities for improvement without being overly negative. Do not include any introduction, summary, or additional headings—only produce the paragraph itself. Ensure the writing flows smoothly and reads naturally, without sounding mechanical or repetitive.\nBelow is the summary of the reviews for this topic: \n"
    return generate_response((prompt + str(content)))
    
def generate_heading(content):
    prompt = "You are a part of a professional Brief-generating agent. Your task is to create a **creative, engaging, and concise heading** that captures the essence of the topic discussed in the provided paragraph. The heading should reflect the **main theme or message** conveyed in the paragraph, while being **professional, insightful, and appropriate** for use in a formal report or presentation. Aim for a **compelling and concise title (max 8 words)** that draws attention and encapsulates the key idea discussed in the paragraph without exaggerating or oversimplifying it.Only return the heading, with no additional content.Below is the paragraph:\n"
    return generate_response((prompt + str(content)))

def generate_pos_improve(content):
    prompt = "You are part of a professional Brief-generating agent. Your task is to generate a **polished, thoughtful paragraph** that provides **insights on how the company could leverage the positive aspects** discussed in the provided content to further improve their product or service. Your response should only be a **single paragraph**. Do not include any introduction, summary, or extra content—only the paragraph itself. If there are **meaningful and relevant insights**, include them in a constructive tone, focusing on **opportunities for growth or ways to enhance the user experience**.\n Avoid assumptions or forced suggestions if the content does not provide clear insights. \nIf no meaningful insights are found, respond with a **balanced statement** that highlights the overall strengths. Write in a **professional, non-pushy tone** that flows smoothly and avoids repetition. \nBelow is the content you need to work from: \n"
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
        # After a bit of fiddling we found these parameters to be good. Might not generalise to other datasets
        summary = process_reviews(reviews, num_clusters=13, top_n=2, min_length=1, max_length=700)

        if "negative" in file_name.lower():
            negative_brief.append(str(generate_negative(summary)))
            headings_neg.append(str(generate_heading(negative_brief[-1])))
        else:
            positive_brief.append(str(generate_positive(summary)))
            headings_pos.append(str(generate_heading(positive_brief[-1])))


actions_neg = generate_neg_improve(" ".join(negative_brief))
actions_pos = generate_neg_improve(" ".join(positive_brief))
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

output_file_path = 'final_report_Approach2.txt'

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(final_report)

print(f"Report written to {output_file_path}")