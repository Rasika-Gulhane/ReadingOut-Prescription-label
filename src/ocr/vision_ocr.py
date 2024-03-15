import re, os
from google.cloud import vision_v1
from google.cloud import vision



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'credentials/peak-castle-416420-afc2d99ff254.json'
client = vision.ImageAnnotatorClient()



class TextExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.client = vision_v1.ImageAnnotatorClient()

    def extract_text(self):
        with open(self.image_path, "rb") as image_file:
            content = image_file.read()

        image = vision_v1.Image(content=content)
        response = self.client.text_detection(image=image)

        texts = response.text_annotations
        data = []

        for text in texts:
            data.append({
                'locale': text.locale,
                'description': text.description
            })

        return data

    @staticmethod
    def clean_text(data):
        cleaned_data = []
        for item in data:
            if item['locale'].lower() == 'en':
                cleaned_text = re.sub(r'[^a-zA-Z0-9\s/-]', '', item['description'])
                cleaned_data.append(cleaned_text)
        # print(cleaned_data[0])
        return cleaned_data
        

# # Example usage
# image_path = "path/to/your/image.jpg"
# text_extractor = TextExtractor(image_path)
# text_data = text_extractor.extract_text()
# cleaned_data = TextExtractor.clean_text(text_data)

# Print the cleaned description

