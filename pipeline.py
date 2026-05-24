# pip install chromadb sentence-transformers numpy
# python pipeline.py serve

import json
import os
import argparse
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import threading
import http.server
import socketserver
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "NESTLE GREEK NEW TESTAMENT 1904.json")
DB_PATH = os.path.join(BASE_DIR, "bible_vector_db") # Created by the script
OUTPUT_MAP_FILE = os.path.join(BASE_DIR, "BookChapterVerseMap.json") # Created by the script
HTML_OUTPUT_FILE = os.path.join(BASE_DIR, "index.html") # Created by the script

TOP_K = 50

# ==========================================
# CORE LOGIC
# ==========================================
def build_vector_db():
    """Phase 1: Create the ChromaDB vector database."""
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print(f"[System] Vector database already exists at {DB_PATH}.")
        return True

    print(f"--- Starting Build Process ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Create collection (re-creates if exists to ensure clean start)
    try:
        client.delete_collection(name="bible_verses")
    except:
        pass
    
    collection = client.create_collection(
        name="bible_verses",
        embedding_function=embedding_fn
    )

    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please ensure the JSON source is in the project root.")
        return False

    ids, documents, metadatas = [], [], []

    print("Parsing JSON and preparing vectors...")
    for book, chapters in bible_data.items():
        for chapter_num, verses in chapters.items():
            for verse_num, text in verses.items():
                verse_id = f"{book}_{chapter_num}_{verse_num}"
                ids.append(verse_id)
                documents.append(text)
                metadatas.append({
                    "book": book,
                    "chapter": str(chapter_num),
                    "verse": str(verse_num),
                    "text": text 
                })

    batch_size = 500
    total_verses = len(ids)
    print(f"Starting ingestion of {total_verses} verses into {DB_PATH}...")

    for i in range(0, total_verses, batch_size):
        end_idx = min(i + batch_size, total_verses)
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"Progress: {end_idx}/{total_verses} verses indexed.")

    print("\nSuccess! Vector database created.")
    return True

def generate_proximity_map():
    """Phase 2: Calculate semantic similarities and save to JSON."""
    if not os.path.exists(OUTPUT_MAP_FILE):
        print(f"[System] Proximity map not found. Checking database...")
        if not build_vector_db():
            return False
    
    # Double check DB exists if map was missing
    if not os.path.exists(DB_PATH):
        print("[Error] Database path missing despite build attempt.")
        return False

    print(f"--- Starting Proximity Map Generation ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="bible_verses")

    print("Gathering all verse metadata and embeddings into memory...")
    all_data = collection.get(include=['embeddings', 'metadatas'])
    
    ids = all_data['ids']
    embeddings = np.array(all_data['embeddings'])
    metadatas = all_data['metadatas']
    total_verses = len(ids)

    print(f"Total verses: {total_verses}")
    print(f"Calculating top {TOP_K} cross-book relations per verse...")

    connections = []
    
    for i in range(total_verses):
        current_id = ids[i]
        current_meta = metadatas[i]
        current_book = current_meta['book']
        current_vec = embeddings[i]

        # Vectorized Cosine Similarity
        dot_products = np.dot(embeddings, current_vec)
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(current_vec)
        similarities = dot_products / norms

        sorted_indices = np.argsort(-similarities)

        found_count = 0
        for idx in sorted_indices:
            if idx == i: continue
            
            target_book = metadatas[idx]['book']
            if target_book == current_book: continue
            if found_count >= TOP_K: break
            
            sim_score = similarities[idx]
            if sim_score < 0.3: break

            connections.append({
                "s": current_id,
                "sb": current_book,
                "t": ids[idx],
                "tb": target_book,
                "sim": round(float(sim_score), 4)
            })
            found_count += 1

        if i % 500 == 0:
            print(f"Processed {i}/{total_verses} verses...")

    print(f"Writing {len(connections)} connections to {OUTPUT_MAP_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_MAP_FILE), exist_ok=True)
    with open(OUTPUT_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(connections, f)

    print("Done.")
    return True

def write_html_file():
    """Phase 3: Generate the index.html file automatically."""
    # Check dependencies for HTML
    if not os.path.exists(OUTPUT_MAP_FILE):
        print("[System] Proximity map missing. Generating map...")
        if not generate_proximity_map():
            return False

    print(f"Generating {HTML_OUTPUT_FILE}...")
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bible Semantic Proximity Map</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; background: #0a0a0a; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }
        #controls { position: absolute; top: 10px; left: 10px; background: rgba(20,20,20,0.85); padding: 15px; border-radius: 8px; z-index: 10; width: 220px; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        #report { position: absolute; bottom: 10px; left: 10px; background: rgba(20,20,20,0.9); padding: 15px; border-radius: 8px; width: 480px; max-height: 70vh; overflow-y: auto; z-index: 10; border: 1px solid #444; box-shadow: 0 -4px 15px rgba(0,0,0,0.5); }
        canvas { cursor: crosshair; display: block; }
        select, input { background: #222; color: #fff; border: 1px solid #444; padding: 6px; margin: 8px 0; width: 100%; border-radius: 4px; }
        .stat { font-weight: bold; color: #00d4ff; }
        .verse-text { font-style: italic; color: #eee; line-height: 1.4; display: block; margin: 5px 0; font-size: 0.9em; }
        .sim-label { color: #00d4ff; font-size: 0.85em; font-weight: bold; }
        .line-count { color: #00d4ff; font-weight: bold; margin-bottom: 10px; display: block; }
        hr { border: 0; border-top: 1px solid #333; margin: 10px 0; }
        label { font-size: 0.85em; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    </style>
</head>
<body>
<div id="controls">
    <label>Filter Book</label>
    <select id="bookFilter"><option value="all">All Books</option></select>
    <div id="bookStats"></div>
    <br>
    <label>Connections (K)</label>
    <input type="number" id="kFilter" value="3" min="1" max="50">
</div>
<div id="report"><div id="reportContent">Loading semantic data...</div></div>
<canvas id="viz"></canvas>
<script>
let width, height, centerX, centerY;
const canvas = document.getElementById('viz');
const ctx = canvas.getContext('2d');
let bibleData, connections, nodes = [], nodeMap = new Map(), filteredConnections = [];
let selectedBook = 'all', topK = 3;

Promise.all([
    d3.json('NESTLE GREEK NEW TESTAMENT 1904.json'),
    d3.json('BookChapterVerseMap.json')
]).then(([bData, cData]) => {
    bibleData = bData;
    connections = cData;
    init();
}).catch(err => console.error("Error loading JSON: Ensure JSON files are in the same folder as this HTML.", err));

function init() {
    const books = Object.keys(bibleData);
    const bookList = d3.select("#bookFilter");
    books.forEach(book => {
        bookList.append("option").attr("value", book).text(book);
        const chapters = Object.keys(bibleData[book]);
        chapters.forEach(ch => {
            const verses = Object.keys(bibleData[book][ch]);
            verses.forEach(vs => {
                const id = `${book}_${ch}_${vs}`;
                const node = { id, book, chapter: ch, verse: vs, text: bibleData[book][ch][vs] };
                nodes.push(node);
                nodeMap.set(id, node);
            });
        });
    });
    window.addEventListener('resize', resize);
    resize();
    d3.select("#bookFilter").on("change", function() { selectedBook = this.value; updateStats(); updateViz(); });
    d3.select("#kFilter").on("input", function() { topK = parseInt(this.value) || 1; updateViz(); });
    updateStats();
    updateViz();
}

function resize() {
    width = window.innerWidth; height = window.innerHeight;
    canvas.width = width; canvas.height = height;
    centerX = width / 2; centerY = height / 2;
    updateViz();
}

function updateStats() {
    const statsDiv = d3.select("#bookStats");
    if (selectedBook === 'all') statsDiv.html("");
    else {
        const book = bibleData[selectedBook];
        const chCount = Object.keys(book).length;
        let vsCount = 0;
        Object.values(book).forEach(c => vsCount += Object.keys(c).length);
        statsDiv.html(`Chapters: <span class="stat">${chCount}</span><br>Verses: <span class="stat">${vsCount}</span>`);
    }
}

function updateViz() {
    ctx.clearRect(0, 0, width, height);
    const radiusLimit = Math.min(width, height) * 0.42;
    const books = Object.keys(bibleData);
    const bookStats = books.map(book => {
        let count = 0;
        const chapters = bibleData[book];
        Object.values(chapters).forEach(ch => count += Object.keys(ch).length);
        return { name: book, verseCount: count };
    });
    const minVerses = Math.min(...bookStats.map(b => b.verseCount));
    let bookSpokeData = [];
    let totalSpokeCount = 0;
    bookStats.forEach(stat => {
        const spokesNeeded = Math.ceil(stat.verseCount / minVerses);
        bookSpokeData.push({ name: stat.name, spokeCount: spokesNeeded, verseCount: stat.verseCount, color: (bookSpokeData.length % 2 === 0) ? "#00d4ff" : "#ffea00" });
        totalSpokeCount += spokesNeeded;
    });
    const spokeAngleStep = (Math.PI * 2) / totalSpokeCount;
    const NUM_RINGS = minVerses; 
    const minRadius = radiusLimit * 0.2; 
    const usableRadius = radiusLimit - minRadius;
    let currentSpokeIndex = 0;
    const bookSpokeOffsets = new Map();
    bookSpokeData.forEach(b => { bookSpokeOffsets.set(b.name, currentSpokeIndex); currentSpokeIndex += b.spokeCount; });
    const bookNodeGroups = d3.group(nodes, d => d.book);

    nodes.forEach(n => {
        const bData = bookSpokeData.find(b => b.name === n.book);
        if (!bData) return;
        const bookVerses = bookNodeGroups.get(n.book);
        const vIdxInBook = bookVerses.findIndex(v => v.id === n.id);
        const spokeWithinBook = vIdxInBook % bData.spokeCount;
        const globalSpokeIndex = bookSpokeOffsets.get(n.book) + spokeWithinBook;
        const angle = globalSpokeIndex * spokeAngleStep;
        const ringIndex = Math.floor((vIdxInBook / bData.verseCount) * NUM_RINGS);
        const r = minRadius + (ringIndex * (usableRadius / Math.max(1, NUM_RINGS - 1)));
        n.x = centerX + r * Math.cos(angle);
        n.y = centerY + r * Math.sin(angle);
        n.bookColor = bData.color;
    });

    let validConnections = [];
    const connectionCountPerVerse = new Map();
    for (let i = 0; i < connections.length; i++) {
        const c = connections[i];
        const sourceNode = nodeMap.get(c.s);
        const targetNode = nodeMap.get(c.t);
        if (!sourceNode || !targetNode) continue;
        const involvesSelectedBook = (selectedBook === 'all') || (sourceNode.book === selectedBook || targetNode.book === selectedBook);
        if (involvesSelectedBook) {
            const sCount = connectionCountPerVerse.get(c.s) || 0;
            const tCount = connectionCountPerVerse.get(c.t) || 0;
            if (sCount < topK && tCount < topK) {
                validConnections.push(c);
                connectionCountPerVerse.set(c.s, sCount + 1);
                connectionCountPerVerse.set(c.t, tCount + 1);
            }
        }
    }

    if (validConnections.length > 0) {
        ctx.lineWidth = 0.6;
        validConnections.forEach(c => {
            const s = nodeMap.get(c.s);
            const t = nodeMap.get(c.t);
            if (s && t) {
                ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y);
                ctx.strokeStyle = "rgba(255, 255, 255, 0.3)"; ctx.stroke();
            }
        });
    }

    nodes.forEach(n => {
        if (selectedBook === 'all' || n.book === selectedBook) {
            ctx.beginPath(); ctx.arc(n.x, n.y, 0.9, 0, Math.PI * 2);
            ctx.fillStyle = n.bookColor || "#ffffff"; ctx.fill();
        }
    });

    ctx.font = "bold 11px Arial"; ctx.textAlign = "center";
    let runningSpokeCount = 0;
    bookSpokeData.forEach(b => {
        let sumAngle = 0;
        for(let i=0; i<b.spokeCount; i++) sumAngle += (runningSpokeCount + i) * spokeAngleStep;
        const avgAngle = sumAngle / b.spokeCount;
        const lx = centerX + (radiusLimit + 30) * Math.cos(avgAngle);
        const ly = centerY + (radiusLimit + 30) * Math.sin(avgAngle);
        ctx.fillStyle = b.color; ctx.fillText(b.name.toUpperCase(), lx, ly);
        runningSpokeCount += b.spokeCount;
    });
    generateReport(validConnections);
}

function generateReport(currentSet) {
    const reportDiv = document.getElementById('reportContent');
    if (currentSet.length === 0) { reportDiv.innerHTML = "No connections to display."; return; }
    const filteredAndSorted = currentSet.filter(c => c.sim < 1).sort((a, b) => b.sim - a.sim);
    if (filteredAndSorted.length === 0) { reportDiv.innerHTML = "No non-identical matches found."; return; }
    let html = `<span class="line-count">Displaying ${filteredAndSorted.length.toLocaleString()} unique matches</span>`;
    html += `<div style="color: #00d4ff; font-size: 0.8em; margin-bottom: 5px; text-transform: uppercase;">Top 20 Most Similar Pairs</div>`;
    const limit = Math.min(filteredAndSorted.length, 20);
    for(let i = 0; i < limit; i++) {
        const c = filteredAndSorted[i];
        const s = nodeMap.get(c.s);
        const t = nodeMap.get(c.t);
        html += `<div style="margin-bottom: 10px;"><span class="sim-label">Similarity: ${c.sim}</span><br><span class="verse-text">"${s.text}"</span><div style="text-align:center; color:#444; font-size:0.8em;">?</div><span class="verse-text">"${t.text}"</span><small style="color:#666;">${s.book} ${s.chapter}:${s.verse} ? ${t.book} ${t.chapter}:${t.verse}</small></div><hr>`;
    }
    const least = filteredAndSorted[filteredAndSorted.length - 1];
    if (least) {
        const ls = nodeMap.get(least.s);
        const lt = nodeMap.get(least.t);
        html += `<div style="color: #ff4444; font-size: 0.8em; margin-top: 10px; text-transform: uppercase;">Least Similar Pair</div><span class="sim-label" style="color:#ff4444">Similarity: ${least.sim}</span><br><span class="verse-text">"${ls.text}"</span><div style="text-align:center; color:#444; font-size:0.8em;">?</div><span class="verse-text">"${lt.text}"</span><small style="color:#666;">${ls.book} ${ls.chapter}:${ls.verse} ? ${lt.book} ${lt.chapter}:${lt.verse}</small>`;
    }
    reportDiv.innerHTML = html;
}
</script>
</body>
</html>"""
    with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML file generated: {HTML_OUTPUT_FILE}")
    return True

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Bible Semantic Proximity Pipeline")
    parser.add_argument("action", choices=["build", "map", "html", "serve"], 
                        help="build: Create DB, map: Create proximity JSON, "
                             "html: Generate index.html, serve: Start server and open browser")
    
    args = parser.parse_args()

    if args.action == "build":
        build_vector_db()
    elif args.action == "map":
        generate_proximity_map()
    elif args.action == "html":
        write_html_file()
    elif args.action == "serve":
        if not write_html_file():
            print("[Error] Failed to generate required files for serving.")
            return

        PORT = 8000
        Handler = http.server.SimpleHTTPRequestHandler

        def run_server():
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"\n[Server] Serving at http://localhost:{PORT}")
                print("[Server] Press Ctrl+C to stop the server.")
                httpd.serve_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        print(f"[Browser] Opening index.html...")
        webbrowser.open(f"http://localhost:{PORT}/index.html")

        try:
            while True:
                server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\n[System] Stopping server...")

if __name__ == "__main__":
    main()
