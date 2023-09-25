import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import open_clip


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

text_embeded = tokenizer(["trying embedding with openclip tokenizer and model"])
print("hello")
print(text_embeded)
model.eval()
def create_image_embedding(image_path):
  try:
    image = Image.open(image_path).convert('RGB')
    input_image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
      embedding = model.encode_image(input_image).reshape(512)
      return embedding.detach().numpy()
  except Exception as e:
    print("Error:", e)
    return None
    
# image_path = "./images/cat1.jpg"
# cat1 = create_image_embedding(image_path)

# # 'embedding' now contains a dense vector representation of the image
# print("Image Embedding Shape:", cat1.shape)
# print("Image Embedding:", cat1)

# image_path = "./images/cat2.jpg"
# cat2 = create_image_embedding(image_path)

# # 'embedding' now contains a dense vector representation of the image
# print("Image Embedding Shape:", cat2.shape)
# print("Image Embedding:", cat2)

# dog1 = create_image_embedding("./images/dog1.jpg")
# dog2 = create_image_embedding("./images/dog2.jpg")
# dog3 = create_image_embedding("./images/dog3.jpg")
# person1 = create_image_embedding("./images/person1.jpg")
# person2 = create_image_embedding("./images/person2.jpg")
# person3 = create_image_embedding("./images/person3.jpg")

# #Imports a PyMilvus package:
# from pymilvus import (
#     connections,
#     utility,
#     FieldSchema,
#     CollectionSchema,
#     DataType,
#     Collection,
# )

# #Connect to the Milvus
# connections.connect("default", host="localhost", port="19530")

# #Creates a collection:
# fields = [
#     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
#     FieldSchema(name="words", dtype=DataType.VARCHAR, max_length=50),
#     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)
# ]
# schema = CollectionSchema(fields, "Simple Demo for image similar search")
# image = Collection("image", schema)

# # Builds indexes on the entities:

# index = {
#     "index_type": "IVF_FLAT",
#     "metric_type": "L2",
#     "params": {"nlist": 128},
# }

# image.create_index("embeddings", index)

# #Insert data in collection
# data = [
#     [1,2,3,4,5,6,7],  # field pk
#     ['cat1','cat2','dog1','dog2','dog3','person1','person2'],  # field words
#     [cat1, cat2, dog1, dog2, dog3, person1, person2],  # field embeddings
# ]

# image.insert(data)
# image.flush()
# image.load()

# search_params = {"metric_type": "L2"}

# results = image.search(
# 	data=[dog1], 
# 	anns_field="embeddings", 
# 	param=search_params,
# 	limit=4, 
# 	expr=None,
# 	# set the names of the fields you want to retrieve from the search result.
# 	output_fields=['words'],
# 	consistency_level="Strong"
# )

# for i in range(0,len(results[0])):
#     name = results[0][i].entity.get('words')
#     print(name)
#     # display(Image.open('./images/'+name+'.jpg'))

# results = image.search(
# 	data=[person3], 
# 	anns_field="embeddings", 
# 	param=search_params,
# 	limit=2, 
# 	expr=None,
# 	# set the names of the fields you want to retrieve from the search result.
# 	output_fields=['words'],
# 	consistency_level="Strong"
# )

# for i in range(0,len(results[0])):
#     name = results[0][i].entity.get('words')
#     print(name)
#     # display(Image.open('./images/'+name+'.jpg'))
