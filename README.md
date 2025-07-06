# CrediTrust RAG-Powered Chatbot

## Project Overview

CrediTrust faced challenges with efficiently handling customer queries and internal data retrieval related to credit scoring and fraud detection. This project implements a Retrieval-Augmented Generation (RAG) chatbot solution to enhance query accuracy by combining document retrieval with large language model (LLM) generation.

The chatbot integrates:

- **Exploratory Data Analysis (EDA)** and preprocessing of internal datasets to ensure clean and relevant information.
- **Text chunking and embedding** using sentence transformers (`all-MiniLM-L6-v2`) to convert documents into vector representations.
- **FAISS index** for efficient similarity search of relevant document chunks.
- A **RAG pipeline** that retrieves top relevant chunks based on user queries, combines them with a prompt template, and uses an LLM (via Hugging Face Inference API) to generate accurate answers.
- An interactive **Gradio UI** for end-user testing and demonstrations.

## Key Components

1. **Data Preprocessing & EDA**  
   - Data cleaning and filtering to remove noise and irrelevant information.  
   - Insights from exploratory analysis to guide chunking and embedding strategies.

2. **Chunking and Embedding**  
   - Texts split into manageable chunks preserving context.  
   - Embeddings created using `all-MiniLM-L6-v2` transformer for semantic similarity.

3. **Retrieval and Generation**  
   - FAISS index built over embeddings for fast similarity retrieval.  
   - Custom prompt template merges query with top chunks to guide LLM responses.

4. **Evaluation**  
   - Qualitative assessment with sample questions, scoring answer quality and source relevance.  
   - Identified strengths and areas for improvement to guide future iterations.

## How to Use

- Clone the repo and install dependencies.
- Set your Hugging Face API token as an environment variable (`HF_API_TOKEN`).
- Run the scripts to build the FAISS index and start the chatbot UI.
- Test sample queries to see the RAG pipeline in action.

## Future Improvements

- Enhance chunking strategies to improve context retention.
- Experiment with larger or domain-specific LLM models.
- Automate evaluation with a larger dataset and metrics.

---

For detailed implementation, evaluation tables, and discussion, refer to the full project report.

