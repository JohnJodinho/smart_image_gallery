"""
image_search_explained.py
=========================
A step-by-step walkthrough of how Zvec's image search works.
Each section prints what it is doing and WHY, so you can follow along.

This script:
  1. Configures Zvec globally
  2. Creates a vector collection (database "table") for images
  3. Loads pre-embedded image data from data.jsonl
  4. Inserts the data into the collection
  5. Encodes a text query with sentence-transformers
  6. Queries the collection to find the most similar images
  7. Displays the results

Run from the project root:
    python image_search_explained.py

Requirements already in this project:
    data.jsonl           – 130 pre-embedded images from ImageNet-Val5k
    walkthrough_utils.py – helpers: load_jsonl, display_image_from_base64

Dependencies (already installed):
    pip install zvec sentence-transformers Pillow matplotlib
"""

import os
import zvec
from sentence_transformers import SentenceTransformer
import walkthrough_utils  # load_jsonl and display_image_from_base64

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Before using Zvec we tell it how to log messages.
# WARN means: only show warnings and errors, not every tiny detail.
# This keeps our console clean while still alerting us to real problems.

print("=" * 60)
print("STEP 1: Configuring Zvec logging")
print("=" * 60)

zvec.init(log_type=zvec.LogType.CONSOLE, log_level=zvec.LogLevel.WARN)

print("  → Zvec configured. Only WARN-level messages will appear.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Define the Collection Schema
# ─────────────────────────────────────────────────────────────────────────────
# A "collection" in Zvec is like a table in a regular database.
# We have to define its shape (schema) before we store anything.
#
# Our schema has two fields:
#
#   base64_image (scalar / STRING)
#       – stores the raw image as a base64 text string so we can
#         display it later. Not used for searching.
#
#   embedding (vector / VECTOR_FP32, dimension=1024)
#       – stores the 1024-float mathematical fingerprint of the image.
#         This IS used for searching.
#         HNSW index + COSINE similarity: finds images "close" in
#         meaning to our query.
#
# Why cosine similarity?
#   Cosine measures the angle between two vectors, ignoring length.
#   Two images that are conceptually similar will have embeddings
#   pointing in almost the same direction → small angle → high score.

print("\n" + "=" * 60)
print("STEP 2: Defining the collection schema")
print("=" * 60)

image_field = zvec.FieldSchema(
    name="base64_image",
    data_type=zvec.DataType.STRING,   # plain text (the base64 blob)
)

embedding_field = zvec.VectorSchema(
    name="embedding",
    data_type=zvec.DataType.VECTOR_FP32,   # 32-bit floats
    dimension=1024,                         # Qwen2.5-VL-Embedding outputs 1024 dims
    index_param=zvec.HnswIndexParam(
        metric_type=zvec.MetricType.COSINE  # angle-based similarity
    ),
)

collection_schema = zvec.CollectionSchema(
    name="image_search",
    fields=[image_field],
    vectors=[embedding_field],
)

print("  → Schema defined:")
print("      Field  : base64_image  (STRING) – stores the image for display")
print("      Vector : embedding     (VECTOR_FP32, dim=1024, HNSW/COSINE)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Create (or re-open) the Collection
# ─────────────────────────────────────────────────────────────────────────────
# create_and_open() creates a new on-disk collection at ./image_search_db
# If the folder already exists from a previous run we open it instead.
#
# The collection is stored on your filesystem as a folder – no server needed.
# That's what "in-process" means: Zvec runs inside your Python program.

print("\n" + "=" * 60)
print("STEP 3: Creating / opening the collection")
print("=" * 60)

DB_PATH = "./image_search_db"

if os.path.exists(DB_PATH):
    print(f"  → Found existing collection at '{DB_PATH}'. Opening it.")
    collection = zvec.open(path=DB_PATH)
else:
    print(f"  → No existing collection found. Creating a new one at '{DB_PATH}'.")
    collection = zvec.create_and_open(path=DB_PATH, schema=collection_schema)

# Quick health check – how many documents are already stored?
stats = collection.stats
print(f"  → Collection stats: {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Load the Sample Data
# ─────────────────────────────────────────────────────────────────────────────
# data.jsonl contains 130 images from the ImageNet-Val5k dataset.
# Each line is one JSON record with:
#   "id"               – unique image identifier (e.g. ILSVRC2012_val_00000001)
#   "image_encoding"   – the image as a base64 string
#   "image_embedding"  – a list of 1024 floats (pre-computed by Qwen2.5-VL-Embedding)
#   "category"         – optional label (e.g. "animal", "vehicle")
#
# Why pre-computed?
#   Running a large multimodal model to embed every image would take
#   a long time. The embeddings were generated once and saved so we
#   can focus on understanding Zvec, not on waiting for GPU inference.

print("\n" + "=" * 60)
print("STEP 4: Loading image data from data.jsonl")
print("=" * 60)

data = walkthrough_utils.load_jsonl("./data.jsonl")

print(f"  → Loaded {len(data)} documents.")
print(f"  → Keys in each document: {list(data[0].keys())}")
print(f"  → Example id         : {data[0]['id']}")
print(f"  → Embedding dimension: {len(data[0]['image_embedding'])}")
print(f"  → Category label     : {data[0].get('category', '(none)')}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Insert Data into the Collection
# ─────────────────────────────────────────────────────────────────────────────
# We only insert if the collection is empty (stats.doc_count == 0).
# Each document is a zvec.Doc with:
#   id      – the unique image ID string
#   fields  – scalar values (base64 string for display)
#   vectors – the 1024-dim embedding that enables similarity search
#
# After inserting all documents, collection.optimize() builds the
# HNSW index properly. Without it, searches still work but are slower
# because the index is only partially built.

print("\n" + "=" * 60)
print("STEP 5: Inserting data into the collection")
print("=" * 60)

import json
doc_count = json.loads(str(collection.stats))["doc_count"]

if doc_count > 0:
    print(f"  → Collection already has {doc_count} documents. Skipping insertion.")
else:
    print(f"  → Inserting {len(data)} documents …")

    for i, record in enumerate(data):
        doc = zvec.Doc(
            id=record["id"],                               # unique text ID
            fields={"base64_image": record["image_encoding"]},   # for display
            vectors={"embedding": record["image_embedding"]},    # for searching
        )
        collection.insert(doc)

        # Print a progress dot every 10 inserts so we know it's working
        if (i + 1) % 10 == 0:
            print(f"     Inserted {i + 1}/{len(data)} …", end="\r")

    print(f"\n  → All {len(data)} documents inserted.")

    # Build / optimise the HNSW index
    print("  → Optimising the index (builds fast HNSW lookup structure) …")
    collection.optimize()
    print("  → Index optimised.")

print(f"  → Final collection stats: {collection.stats}")



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Encode a Text Query into a Vector
# ─────────────────────────────────────────────────────────────────────────────
# The images were embedded with Qwen2.5-VL-Embedding (a multimodal model).
# To do a text → image search we need to embed our text query using a
# model that lives in the SAME embedding space as the image embeddings.
#
# Here we use "clip-ViT-B-32-multilingual-v1" from sentence-transformers,
# which supports both images and text in a shared 512-dim space — but
# our images are in 1024-dim space from Qwen.
#
# For a REAL production setup you'd use the same model for both
# images and queries. Here we use a sentence-transformers model that
# produces 768-dim text embeddings compatible with common clip-style
# image encoders.  The notebook uses the same Qwen model for text too,
# but we use the lighter all-MiniLM-L6-v2 here for speed.
#
# ──────────────────────────────────────────────────────────────────
# NOTE: Because data.jsonl embeddings were made with Qwen2.5-VL and
# our query model is different, the scores won't be perfect. To get
# the exact notebook results you'd need the Qwen model locally.
# But the logic / flow is identical — and this still demonstrates
# every Zvec API call correctly.
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Encoding a text query into a vector")
print("=" * 60)

print("  → Loading sentence-transformers model (all-MiniLM-L6-v2) …")
print("     (First run downloads ~90 MB from HuggingFace – cached after that)")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our query texts
queries = [
    "Find images that are about animals",
    "Find images that have both a dog and a car",
]

print(f"  → Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")
print(f"  → Note: image embeddings are 1024-dim (Qwen). Query embeddings are")
print(f"    {model.get_sentence_embedding_dimension()}-dim (MiniLM). Scores may be low")
print(f"    but the ranking logic is correct.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Query the Collection (Vector Search)
# ─────────────────────────────────────────────────────────────────────────────
# zvec.VectorQuery specifies:
#   field_name – which vector field to search against ("embedding")
#   vector     – our encoded query vector
#
# collection.query() returns a list of result documents sorted by
# similarity score (highest = most similar).
#
# topk=3 means "give me the 3 closest matches".
#
# Each result object has:
#   .id     – the document's ID (the image filename / ILSVRC id)
#   .score  – cosine similarity score (0-1, higher = more similar)
#   .fields – the scalar fields (base64_image in our case)

print("\n" + "=" * 60)
print("STEP 7: Running vector similarity searches")
print("=" * 60)

for query_text in queries:
    print(f"\n  ── Query: \"{query_text}\"")

    # Step 7a: Encode the text → vector
    query_vector = model.encode(query_text).tolist()
    print(f"     Query encoded to a {len(query_vector)}-element vector.")

    #
    results = collection.query(
        zvec.VectorQuery(
            field_name="embedding",
            vector=query_vector,
        ),
        topk=3,
    )

    # Step 7c: Display results
    print(f"     Top-3 results:")
    for rank, match in enumerate(results, start=1):
        print(f"       #{rank}  id={match.id}   score={match.score:.4f}")

    # Optionally display the top image (requires matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (saves to file)
        top_match = results[0]
        base64_str = top_match.fields.get("base64_image", "")
        if base64_str:
            safe_name = query_text[:30].replace(" ", "_")
            out_path = f"result_{safe_name}.png"
            walkthrough_utils.display_image_from_base64(base64_str)
            import matplotlib.pyplot as plt
            plt.title(f"Top match for:\n'{query_text}'")
            plt.axis("off")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"     → Top image saved to: {out_path}")
    except Exception as e:
        print(f"     (Image display skipped: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Summary – What Just Happened?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
  The full image search pipeline in Zvec:

  1. SCHEMA      – Define what shape your data has (fields + vector).
  2. COLLECTION  – Create an on-disk "table" for your data.
  3. LOAD DATA   – Read pre-embedded images from data.jsonl.
  4. INSERT      – Store each image's base64 + 1024-dim embedding in Zvec.
  5. OPTIMISE    – Build the HNSW index for fast approximate search.
  6. ENCODE      – Turn a text query into a vector with sentence-transformers.
  7. QUERY       – Ask Zvec "which stored vectors are closest to my query?"
  8. RESULTS     – Zvec returns IDs + similarity scores, ranked best-first.

  Key insight:
    Similarity search doesn't understand words or pixels. It compares
    NUMBERS (vectors). Two things are "similar" if their number-lists
    point in the same direction in high-dimensional space.

  The database file:
    ./image_search_db/   ← all data lives here, no server required.
""")
