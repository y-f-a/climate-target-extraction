# Climate Target Extraction 

This repository is the home to a set of controlled experiments comparing GPT-5, GPT-5.1, and GPT-5.2 on climate target extraction from corporate disclosures, including using a RAG pipeline.

Specifically, it is a controlled, quantitative comparison of GPT-5, GPT-5.1, and GPT-5.2 for extracting SBTi-aligned climate emissions targets from corporate disclosures. The evaluation focuses on near-term targets (e.g. 2030) and net-zero commitments (e.g. 2050), and excludes other categories of climate targets. All model variants are evaluated under an identical retrieval-augmented generation (RAG) setup to isolate the effect of model choice.

---

## Data

The evaluation uses annual reports and sustainability reports from 7 companies across 2 reporting years, totaling approximately 650 MB of documents.

To avoid redistribution and licensing issues, I did not include source documents in this repository. The notebook expects documents to be provided locally in the format described within the notebook.
