import weaviate
from weaviate.classes.config import Configure, DataType, Property

# Connect to Weaviate
client = weaviate.connect_to_local()

# Create a collection
client.collections.create(
    name="Article",
    properties=[Property(name="content", data_type=DataType.TEXT)],
    vector_config=Configure.Vectors.text2vec_model2vec(),  # Use a vectorizer to generate embeddings during import
    # vector_config=Configure.Vectors.self_provided()  # If you want to import your own pre-generated embeddings
)

# Insert objects and generate embeddings
articles = client.collections.get("Article")
articles.data.insert_many(
    [
        {"content": "Vector databases enable semantic search"},
        {"content": "Machine learning models generate embeddings"},
        {"content": "Weaviate supports hybrid search capabilities"},
    ]
)

# Perform semantic search
results = articles.query.near_text(query="Search objects by meaning", limit=1)
print(results.objects[0])

client.close()