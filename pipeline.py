# pip install chromadb sentence-transformers numpy
# python pipeline.py

import json
import os
import argparse
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

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
    print(f"--- Starting Build Process ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        client.delete_collection(name="bible_verses")
    except:
        pass
    
    collection = client.create_collection(
        name="bible_verses",
        embedding_function=embedding_fn
    )

    print(f"Loading {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please ensure the JSON source is in the project root.")
        return False

    ids, documents, metadatas = [], [], []

    print("Parsing JSON and preparing vectors")
    for book, chapters in bible_data.items():
        for chapter_num, verses in chapters.items():
            chapter_text_parts = []
            
            for verse_num, text in verses.items():
                verse_id = f"{book}_{chapter_num}_{verse_num}"
                ids.append(verse_id)
                documents.append(text)
                metadatas.append({
                    "book": book,
                    "chapter": str(chapter_num),
                    "verse": str(verse_num),
                    "text": text,
                    "type": "verse"
                })
                chapter_text_parts.append(text)
            
            chapter_id = f"{book}_{chapter_num}"
            chapter_full_text = " ".join(chapter_text_parts)
            ids.append(chapter_id)
            documents.append(chapter_full_text)
            metadatas.append({
                "book": book,
                "chapter": str(chapter_num),
                "verse": "0", 
                "text": chapter_full_text,
                "type": "chapter"
            })

    batch_size = 500
    total_entries = len(ids)
    print(f"Starting ingestion of {total_entries} entries (verses + chapters) into {DB_PATH}")

    for i in range(0, total_entries, batch_size):
        end_idx = min(i + batch_size, total_entries)
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"Progress: {end_idx}/{total_entries} entries indexed.")

    print("\nSuccess! Vector database created.")
    return True

def generate_proximity_map():
    """Phase 2: Calculate semantic similarities and save to JSON."""
    if not os.path.exists(DB_PATH):
        print(f"[System] Database not found. Triggering build")
        if not build_vector_db():
            return False
    
    print(f"--- Starting Proximity Map Generation ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="bible_verses")

    print("Gathering all metadata and embeddings into memory")
    all_data = collection.get(include=['embeddings', 'metadatas'])
    
    ids = all_data['ids']
    embeddings = np.array(all_data['embeddings'])
    metadatas = all_data['metadatas']
    total_entries = len(ids)

    print(f"Total entries: {total_entries}")
    print(f"Calculating top {TOP_K} cross-book relations per entry")

    connections = []
    
    for i in range(total_entries):
        current_id = ids[i]
        current_meta = metadatas[i]
        current_book = current_meta['book']
        current_vec = embeddings[i]

        dot_products = np.dot(embeddings, current_vec)
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(current_vec)
        similarities = dot_products / norms

        sorted_indices = np.argsort(-similarities)

        found_count = 0
        for idx in sorted_indices:
            if idx == i: continue
            
            target_meta = metadatas[idx]
            target_book = target_meta['book']
            
            if target_book == current_book: continue
            if found_count >= TOP_K: break
            
            sim_score = similarities[idx]
            if sim_score < 0.3: break

            connections.append({
                "s": current_id,
                "sb": current_book,
                "t": ids[idx],
                "tb": target_book,
                "sim": round(float(sim_score), 4),
                "stype": current_meta.get('type', 'verse'),
                "ttype": target_meta.get('type', 'verse')
            })
            found_count += 1

        if i % 500 == 0:
            print(f"Processed {i}/{total_entries} entries")

    print(f"Writing {len(connections)} connections to {OUTPUT_MAP_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_MAP_FILE), exist_ok=True)
    with open(OUTPUT_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(connections, f)

    print("Done.")
    return True

def write_html_file():
    """Phase 3: Generate JS data files and an HTML file that loads them as scripts."""
    if not os.path.exists(OUTPUT_MAP_FILE):
        print("[System] Proximity map missing. Generating map")
        if not generate_proximity_map():
            return False

    print(f"Generating assets if needed")

    try:
        with open("NESTLE GREEK NEW TESTAMENT 1904.json", 'r', encoding='utf-8') as f:
            bible_content = f.read()
        with open(os.path.join(BASE_DIR, "bible_data.js"), 'w', encoding='utf-8') as f:
            f.write(f"const bibleData = {bible_content};")
        print("Created bible_data.js")
    except Exception as e:
        print(f"Error creating bible_data.js: {e}")
        return False

    try:
        with open(OUTPUT_MAP_FILE, 'r', encoding='utf-8') as f:
            map_content = f.read()
        with open(os.path.join(BASE_DIR, "map_data.js"), 'w', encoding='utf-8') as f:
            f.write(f"const connections = {map_content};")
        print("Created map_data.js")
    except Exception as e:
        print(f"Error creating map_data.js: {e}")
        return False

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Bible Semantic Proximity Map</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="bible_data.js"></script>
    <script src="map_data.js"></script>
    <style>
        body { 
            margin: 0; 
            background: #0a0a0a; 
            color: #e0e0e0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            overflow: hidden; 
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }
        
        #viz-container {
            position: relative;
            flex-grow: 1;
            width: 100%;
            overflow: hidden;
            touch-action: none;
        }

        #controls, #report { 
            background: rgba(20,20,20,0.95); 
            border: 1px solid #333; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.5); 
            z-index: 10; 
            box-sizing: border-box;
        }
        
        /* Desktop Styles */
        @media (min-width: 768px) {
            #controls { position: absolute; top: 10px; left: 10px; padding: 15px; border-radius: 8px; width: 220px; }
            #report { position: absolute; bottom: 10px; left: 10px; padding: 15px; border-radius: 8px; width: 480px; max-height: 40vh; overflow-y: auto; }
        }

        /* Mobile Styles */
        @media (max-width: 767px) {
            #controls { width: 100%; padding: 10px; border-radius: 0; }
            #report { width: 100%; max-height: 35vh; border-radius: 0; }
            .panel-header { font-size: 0.9em; }
        }

        .panel-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            cursor: pointer; 
            user-select: none;
            margin-bottom: 10px;
        }
        .panel-header:hover { color: #00d4ff; }
        .collapsed .panel-content { display: none; }
        .collapsed { padding: 8px 15px !important; }
        .collapsed .panel-header { margin-bottom: 0; }

        canvas { cursor: crosshair; display: block; touch-action: none; }
        select, input { 
            background: #222; 
            color: #fff; 
            border: 1px solid #444; 
            padding: 12px; /* Larger tap target for mobile */
            margin: 8px 0; 
            width: 100%; 
            border-radius: 4px; 
            font-size: 16px; /* Prevents iOS zoom on focus */
        }
        .stat { font-weight: bold; color: #00d4ff; }
        .verse-text { font-style: italic; color: #eee; line-height: 1.4; display: block; margin: 5px 0; font-size: 0.9em; white-space: pre-wrap; }
        .sim-label { color: #00d4ff; font-size: 0.85em; font-weight: bold; }
        .line-count { color: #00d4ff; font-weight: bold; margin-bottom: 10px; display: block; }
        hr { border: 0; border-top: 1px solid #333; margin: 10px 0; }
        label { font-size: 0.85em; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
        .hidden { display: none; }
        .selected-node-info { color: #ff00ff; font-size: 0.8em; margin-bottom: 10px; font-weight: bold; }
    </style>
</head>
<body>

<div id="controls">
    <div class="panel-header" onclick="togglePanel('controls')">
        <label>Filters</label>
        <span>&#x25BC;</span>
    </div>
    <div class="panel-content">
        <label>Filter Book</label>
        <select id="bookFilter"><option value="all">All Books</option></select>
        
        <div id="drilldownContainer" class="hidden">
            <label>Drill Down: Chapter</label>
            <select id="chapterFilter"><option value="all">All Chapters</option></select>
        </div>

        <div id="bookStats"></div>
        <hr>
        <label>View Mode</label>
        <select id="viewMode">
            <option value="all">All (Mixed)</option>
            <option value="chapter">Chapters Only</option>
            <option value="verse">Verses Only</option>
        </select>
        <hr>
        <label>Connections (K)</label>
        <input type="number" id="kFilter" value="3" min="1" max="50">
    </div>
</div>

<div id="viz-container">
    <canvas id="viz"></canvas>
</div>

<div id="report">
    <div class="panel-header" onclick="togglePanel('report')">
        <label>Semantic Report</label>
        <span>&#x25BC;</span>
    </div>
    <div class="panel-content">
        <div id="reportContent">Loading semantic data...</div>
    </div>
</div>

<script>
let width, height, centerX, centerY;
const canvas = document.getElementById('viz');
const ctx = canvas.getContext('2d');
let nodes = [], nodeMap = new Map(), filteredConnections = [];
let selectedBook = 'all', selectedChapter = 'all', viewMode = 'all', topK = 3;
let activeNodeId = null;

function togglePanel(id) {
    document.getElementById(id).classList.toggle('collapsed');
}

function init() {
    if (typeof bibleData === 'undefined' || typeof connections === 'undefined') {
        document.getElementById('reportContent').innerHTML = "Error: Data files not found.";
        return;
    }

    const books = Object.keys(bibleData);
    const bookList = d3.select("#bookFilter");
    
    books.forEach(book => {
        bookList.append("option").attr("value", book).text(book);
        const chapters = Object.keys(bibleData[book]);
        chapters.forEach(ch => {
            const chapterVerses = bibleData[book][ch];
            const chapterFullText = Object.values(chapterVerses).join(" ");
            const chapterId = `${book}_${ch}`;
            const chapterNode = { id: chapterId, book, chapter: ch, type: 'chapter', text: chapterFullText };
            nodes.push(chapterNode);
            nodeMap.set(chapterId, chapterNode);

            Object.keys(chapterVerses).forEach(vs => {
                const id = `${book}_${ch}_${vs}`;
                const node = { id, book, chapter: ch, verse: vs, text: chapterVerses[vs], type: 'verse' };
                nodes.push(node);
                nodeMap.set(id, node);
            });
        });
    });

    d3.select("#bookFilter").on("change", function() { 
        selectedBook = this.value; 
        updateChapterDropdown();
        updateStats(); 
        activeNodeId = null;
        updateViz(); 
    });

    d3.select("#chapterFilter").on("change", function() {
        selectedChapter = this.value;
        activeNodeId = null;
        updateViz();
    });

    d3.select("#viewMode").on("change", function() {
        viewMode = this.value;
        activeNodeId = null;
        updateViz();
    });

    d3.select("#kFilter").on("input", function() { 
        topK = parseInt(this.value) || 1; 
        updateViz(); 
    });

    // Handle both Mouse and Touch
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('touchstart', function(e) {
        // Prevent scrolling while interacting with the map
        if (e.touches.length === 1) {
            handleCanvasClick(e.changedTouches[0]);
        }
    }, {passive: true});

    window.addEventListener('resize', resize);
    resize();
    updateChapterDropdown();
    updateStats();
    updateViz();
}

function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let clickedNode = null;
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        const isVisible = (selectedBook === 'all' || n.book === selectedBook) &&
                          (selectedChapter === 'all' || n.chapter === selectedChapter) &&
                          (viewMode === 'all' || (viewMode === 'chapter' ? n.type === 'chapter' : n.type === 'verse'));
        
        if (isVisible) {
            const dx = mouseX - n.x;
            const dy = mouseY - n.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const radius = n.type === 'chapter' ? 10 : 6; // Increased hit area for mobile
            if (dist < radius) {
                clickedNode = n;
                break;
            }
        }
    }

    if (clickedNode) {
        activeNodeId = clickedNode.id;
    } else {
        activeNodeId = null;
    }
    updateViz();
}

function updateChapterDropdown() {
    const chapterList = d3.select("#chapterFilter");
    const drilldownContainer = document.getElementById('drilldownContainer');
    chapterList.html('<option value="all">All Chapters</option>');
    if (selectedBook === 'all') {
        drilldownContainer.classList.add('hidden');
        selectedChapter = 'all';
    } else {
        drilldownContainer.classList.remove('hidden');
        selectedChapter = 'all';
        Object.keys(bibleData[selectedBook]).forEach(ch => {
            chapterList.append("option").attr("value", ch).text(`Chapter ${ch}`);
        });
    }
}

function resize() {
    const container = document.getElementById('viz-container');
    width = container.clientWidth; 
    height = container.clientHeight;
    
    // Handle High DPI screens
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);

    centerX = width / 2; 
    centerY = height / 2;
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
        let vIdxInBook;
        const bookVerses = bookNodeGroups.get(n.book).filter(v => v.type === 'verse');
        if (n.type === 'verse') {
            vIdxInBook = bookVerses.findIndex(v => v.id === n.id);
        } else {
            const firstVerse = bookVerses.find(v => v.chapter === n.chapter);
            vIdxInBook = bookVerses.indexOf(firstVerse);
        }
        const spokeWithinBook = vIdxInBook % bData.spokeCount;
        const globalSpokeIndex = bookSpokeOffsets.get(n.book) + spokeWithinBook;
        const angle = globalSpokeIndex * spokeAngleStep;
        const ringIndex = Math.floor((vIdxInBook / bData.verseCount) * NUM_RINGS);
        const r = minRadius + (ringIndex * (usableRadius / Math.max(1, NUM_RINGS - 1)));
        n.x = centerX + r * Math.cos(angle);
        n.y = centerY + r * Math.sin(angle);
        n.bookColor = bData.color;
    });

    const visibleNodes = nodes.filter(n => {
        const bookMatch = (selectedBook === 'all' || n.book === selectedBook);
        const chapterMatch = (selectedChapter === 'all' || n.chapter === selectedChapter);
        let typeMatch = true;
        if (viewMode === 'chapter') typeMatch = (n.type === 'chapter');
        else if (viewMode === 'verse') typeMatch = (n.type === 'verse');
        return bookMatch && chapterMatch && typeMatch;
    });

    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    let validConnections = [];
    const connectionCountPerVerse = new Map();
    for (let i = 0; i < connections.length; i++) {
        const c = connections[i];
        const matchesSelection = activeNodeId ? (c.s === activeNodeId || c.t === activeNodeId) : true;
        
        if ( ( visibleNodeIds.has(c.s) || visibleNodeIds.has(c.t) ) && matchesSelection) {
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

    visibleNodes.forEach(n => {
        ctx.beginPath(); 
        if (n.type === 'chapter') {
            ctx.arc(n.x, n.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = "#ff00ff"; 
        } else {
            ctx.arc(n.x, n.y, 1, 0, Math.PI * 2);
            ctx.fillStyle = n.bookColor || "#ffffff"; 
        }
        if (n.id === activeNodeId) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = "#fff";
            ctx.arc(n.x, n.y, n.type === 'chapter' ? 5 : 2.5, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.shadowBlur = 0;
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

    generateReport(validConnections, activeNodeId);
}

function generateReport(currentSet, filterId) {
    const reportDiv = document.getElementById('reportContent');
    if (currentSet.length === 0) { reportDiv.innerHTML = "No connections to display."; return; }
    
    const filteredAndSorted = currentSet.filter(c => c.sim < 1).sort((a, b) => b.sim - a.sim);
    if (filteredAndSorted.length === 0) { reportDiv.innerHTML = "No non-identical matches found."; return; }
    
    let html = "";
    if (filterId) {
        const fn = nodeMap.get(filterId);
        html += `<div class="selected-node-info">Showing connections for: ${fn.book} ${fn.type === 'chapter' ? 'Ch ' + fn.chapter : fn.chapter + ':' + fn.verse}</div>`;
    }

    html += `<span class="line-count">Displaying ${filteredAndSorted.length.toLocaleString()} unique matches</span>`;
    html += `<div style="color: #00d4ff; font-size: 0.8em; margin-bottom: 5px; text-transform: uppercase;">Top 20 Most Similar Pairs</div>`;
    
    const limit = Math.min(filteredAndSorted.length, 20);
    for(let i = 0; i < limit; i++) {
        const c = filteredAndSorted[i];
        const s = nodeMap.get(c.s);
        const t = nodeMap.get(c.t);
        const sLabel = s.type === 'chapter' ? `Chapter ${s.chapter}` : `${s.chapter}:${s.verse}`;
        const tLabel = t.type === 'chapter' ? `Chapter ${t.chapter}` : `${t.chapter}:${t.verse}`;
        html += `<div style="margin-bottom: 10px;"><span class="sim-label">Similarity: ${c.sim}</span><br><span class="verse-text">${s.text}</span><div style="text-align:center; color:#444; font-size:0.8em;">?</div><span class="verse-text">${t.text}</span><small style="color:#666;">${s.book} ${sLabel} ? ${t.book} ${tLabel}</small></div><hr>`;
    }
    reportDiv.innerHTML = html;
}

init();
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
    parser.add_argument("action", nargs='?', default="html", choices=["build", "map", "html"], 
                        help="build: Create DB, map: Create proximity JSON, html: Generate index.html (default)")
    args = parser.parse_args()

    if args.action == "build":
        build_vector_db()
    elif args.action == "map":
        generate_proximity_map()
    elif args.action == "html":
        if not write_html_file():
            print("[Error] Failed to generate required files.")
            return

if __name__ == "__main__":
    main()
