import zvec
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

# 1. Load the AI Model (CLIP)
# This model translates both images and text into 512-dimensional vectors
print("Loading CLIP model...")
model = SentenceTransformer("clip-ViT-B-32")

# 2. Define the Database Schema
print("Setting up Zvec database...")
collection_schema = zvec.CollectionSchema(
    name="my_smart_gallery",
    fields=[
        zvec.FieldSchema(
            name="year_added",
            data_type=zvec.DataType.INT32,
            index_param=zvec.InvertIndexParam(enable_range_optimization=True),
        ),
    ],
    vectors=[
        zvec.VectorSchema(
            name="embedding",
            data_type=zvec.DataType.VECTOR_FP32,
            dimension=512,  # CLIP outputs 512 dimensions
            index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
        ),
    ],
)

# Create/Open the collection locally
collection = zvec.create_and_open(
    path="./gallery_db",
    schema=collection_schema,
)


# 3. Add Images to the Database (The "Indexing" Phase)
def index_images(image_directory):
    print(f"Scanning {image_directory} for images...")

    # In a real app, you'd loop through a directory of images
    # For this example, let's pretend we have a list of image paths
    mock_image_paths = ["image1.jpeg", "image2.jpeg"]

    for img_path in mock_image_paths:
        try:
            # Load image and convert to vector
            img = Image.open(img_path)
            vector = model.encode(img).tolist()

            # Insert into Zvec. We use the file path as the unique ID!
            collection.insert(
                zvec.Doc(
                    id=img_path,
                    vectors={"embedding": vector},
                    fields={"year_added": 2026},
                )
            )
            print(f"Indexed: {img_path}")
        except FileNotFoundError:
            print(
                f"Skipping {img_path} (File not found. Create dummy images to test this!)"
            )

    # Build the fast-search index
    collection.optimize()


# 4. Search the Database (The "Querying" Phase)
def search_gallery(text_query):
    print(f"\nSearching gallery for: '{text_query}'")

    # Convert the user's text search into a vector
    text_vector = model.encode(text_query).tolist()

    # Query Zvec for the most visually similar images
    result = collection.query(
        zvec.VectorQuery(
            field_name="embedding",
            vector=text_vector,
        ),
        topk=3,  # Get top 3 matches
    )

    # Print the results
    print("--- Search Results ---")
    for match in result:
        print(f"File: {match.id} | Confidence Score: {match.score:.4f}")


# --- Run the App ---
if __name__ == "__main__":
    # To test this for real, put 3 actual image files in the same folder
    # and name them exactly as they are in the mock_image_paths list.
    index_images("./my_photos")

    # Test a search!
    search_gallery("a fast vehicle")
    search_gallery("an animal resting indoors")
