from azure.cognitiveservices.vision.customvision import training
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient 
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials
from msrest.exceptions import HttpOperationError

import os, time, uuid

cv_endpoint = "https://northeurope.api.cognitive.microsoft.com/"
training_key = "7e166980024f433193e7e27ea6f1baab"
training_images = "FRUIT-16K"

credentials = ApiKeyCredentials(in_headers={"training-key": training_key})
trainer = CustomVisionTrainingClient(endpoint=cv_endpoint, credentials=credentials)

#Wyświetlamy listę wszystkich domenów, żeby zdecydować nad tym, jaki będzie pasował do naszych wymagań   
for domain in trainer.get_domains():   
   print(domain.id, "\t", domain.name)

project = trainer.create_project("FreshFoodDetection - v1","c151d5b5-dd07-472a-acc8-15d29dea8518")

list_of_images = []
dir = os.listdir(training_images)
for tagName in dir:
  tag = trainer.create_tag(project.id, tagName)
  images = os.listdir(os.path.join(training_images,tagName))
  for img in images:
   with open(os.path.join(training_images,tagName,img), "rb") as image_contents:
    list_of_images.append(ImageFileCreateEntry(name=img, contents=image_contents.read(), tag_ids=[tag.id]))
    
  


# Create chunks of 64 images
def chunks(l, n):
 	for i in range(0, len(l), n):
 		yield l[i:i + n]
batchedImages = chunks(list_of_images, 64)

# Upload the images in batches of 64 to the Custom Vision Service

for i in range(0, len(list_of_images), 64):
    try:
     upload_result = trainer.create_images_from_files(project.id, batch=ImageFileCreateBatch(images = list_of_images[i:i + 64], tag_ids=[tag.id]))
     
    except HttpOperationError as e:
     print(e.response.text)
     exit(-1)
    print("Wait...")
 	
  
# Train the model
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
 	iteration = trainer.get_iteration(project.id, iteration.id)
 	print ("Training status: " + iteration.status)
 	time.sleep(1)

# Publish the iteration of the model
publish_iteration_name = '<INSERT ITERATION NAME>'
resource_identifier = '<INSERT RESOURCE IDENTIFIER>'
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, resource_identifier)