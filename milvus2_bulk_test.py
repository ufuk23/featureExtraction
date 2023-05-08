# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection

import random
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="10.1.37.185", port="19530")

#a = [random.randint(1, 200000) for _ in range(200000)]
#print(a)

# utility.drop_collection("artifact")
# utility.drop_collection("artifact")

has = utility.has_collection("artifact")
print(f"Does collection artifact exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |    Int64   |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="artifact_type", dtype=DataType.INT64),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2048)
]

schema = CollectionSchema(fields, "artifact")

print(fmt.format("Create collection `artifact`"))
artifact = Collection("artifact", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `artifact`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.
print(fmt.format("Start inserting entities"))
num_entities = 100
index = num_entities
id = 0

for k in range(50):

    entities = [
        # provide the pk field because `auto_id` is set to False
        [i for i in range(id, index)],
        [random.randrange(1, 5) for _ in range(num_entities)],  # field artifact_type
        [[round(random.random(), 2) for _ in range(2048)] for _ in range(num_entities)],  # field embeddings
    ]

    print([i for i in range(id, index)])
    print([random.randrange(1, 5) for _ in range(num_entities)])
    id = index
    index = index + num_entities
    insert_result = artifact.insert(entities)
    # print(f"Number of entities in Milvus: {artifact.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for artifact collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_SQ8"))
index = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

#artifact.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `artifact` into memory.
print(fmt.format("Start loading"))
artifact.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start artifact searching based on vector similarity"))
vectors_to_search = entities[-1][-2:]
search_params = {
    "metric_type": "l2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = artifact.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["artifact_type"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, artifact_type field: {hit.entity.get('artifact_type')}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `artifact_type > 2`"))

start_time = time.time()
result = artifact.query(expr="artifact_type > 2", output_fields=["artifact_type", "embeddings"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# hybrid search
print(fmt.format("Start hybrid searching with `artifact_type > 3`"))

start_time = time.time()
result = artifact.search(vectors_to_search, "embeddings", search_params, limit=3, expr="artifact_type > 3", output_fields=["artifact_type"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, artifact_type field: {hit.entity.get('artifact_type')}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
ids = insert_result.primary_keys
expr = f"pk in [{ids[0]}, {ids[1]}]"
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = artifact.query(expr=expr, output_fields=["artifact_type", "embeddings"])
print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

# artifact.delete(expr)

result = artifact.query(expr=expr, output_fields=["artifact_type", "embeddings"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the artifact.collection
# print(fmt.format("Drop collection `artifact`"))