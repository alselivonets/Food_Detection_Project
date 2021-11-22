from azure.cognitiveservices.vision.customvision import training
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient 
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry

cv_endpoint = "https://northeurope.api.cognitive.microsoft.com/"
training_key = "7e166980024f433193e7e27ea6f1baab"
training_images = "FRUIT-16K"


trainer = CustomVisionTrainingClient(training_key, endpoint= cv_endpoint)

for domain in trainer.get_domains():   
   print(domain.id, "\t", domain.name)