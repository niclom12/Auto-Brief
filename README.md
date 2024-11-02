# Spotify Review Brief Generator

This project generates a detailed brief from a dataset of Spotify reviews. It leverages natural language processing techniques to cluster reviews, select representative ones, and generate concise summaries.

## Breif Generation section

### Requirements

To run the brief generation segment, follow these steps:

#### Set Up a Virtual Environment
Create a virtual environment for your project to manage dependencies:
```bash
python -m venv my_env
.\my_env\Scripts\activate
pip install -r requirements.txt
```

### Set Up API Keys
Create a .env file in the root directory of the project. Add your API keys for both Hugging Face and Groq:

```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```

### Run the Python Programs
Once the virtual environment is activated and the dependencies are installed, you can run the Python scripts to generate briefs from the Spotify reviews dataset:
```bash
python BriefGenApproach1.py
python BriefGenApproach2.py
```
## Authors 

This repository was created by Nicholas Lombard and Ben Morton.
