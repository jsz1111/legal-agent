import os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from pymilvus import MilvusClient
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")
client = MilvusClient(uri=f"http://{host}:{port}")
for col in client.list_collections():
    stats = client.get_collection_stats(col)
    print(f"{col}: {stats['row_count']} rows")
