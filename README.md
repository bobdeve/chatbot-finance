# chatbot-finance
📊 Exploratory Data Analysis Summary
We began with over 9.6 million complaint records from the CFPB dataset. However, only about 31% of complaints included a narrative, which is essential for language-based analysis. After filtering to include only relevant product categories — Credit Cards, Savings Accounts, and Money Transfers — and removing empty or non-informative entries, we were left with 318,000+ clean narratives.

Analysis of narrative length revealed that while the average word count was moderate, there is a significant number of extremely short complaints (under 10 words), which may not be useful for deep semantic understanding. A small percentage also had very long and detailed complaints (over 250 words), providing rich context for embedding-based retrieval.

These cleaned and filtered narratives form the basis for the next step in our pipeline: generating embeddings and building the Retrieval-Augmented Generation (RAG) system to support internal teams in understanding key user pain points.

