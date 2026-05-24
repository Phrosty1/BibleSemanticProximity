# BibleSemanticProximity

A semantic analysis tool that uses vector embeddings to find connections between different books of the New Testament based on textual similarity.

## Features
- **Vector Database**: Uses ChromaDB and `all-MiniLM-L6-v2` to index the Bible.
- **Semantic Mapping**: Calculates cross-book similarities using cosine similarity.
- **Interactive Visualization**: A D3.js-powered HTML interface to explore connections visually.

## Prerequisites
- Python 3.8+
- `NESTLE GREEK NEW TESTAMENT 1904.json` (must be in the root directory)

## Installation
Install the required dependencies:
```bash
pip install chromadb sentence-transformers numpy
```

## Usage
The pipeline is designed to be "self-healing." You can run the highest-level command, and it will automatically build all necessary components.

### 1. The "Easy" Way (Run everything)
To build the database, generate the similarity map, create the HTML file, and start a local web server:
```bash
python pipeline.py serve
```

### 2. Granular Control
If you want to run specific parts of the pipeline:

- **Build only the Vector DB**:
  ```bash
  python pipeline.py build
  ```
- **Generate the Similarity Map** (requires DB):
  ```bash
  python pipeline.py map
  ```
- **Generate the HTML file** (requires Map):
  ```bash
  python pipeline.py html
  ```

## Project Structure
- `bible_vector_db/`: Local ChromaDB storage (ignored by git).
- `BookChapterVerseMap.json`: The calculated semantic connections.
- `index.html`: The interactive visualization dashboard.
- `NESTLE GREEK NEW TESTAMENT 1904.json`: The source text.
```