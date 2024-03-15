
# from src.llm_model.gpt3 import generate_gpt3_response

from src.ocr.vision_ocr import TextExtractor
import os
from src.llm_model.langchain import LangChainSetup



# image_path = 'dataset/Prescription_12.png'
# folder_path = 'dataset'


# if __name__ == "__main__":

#     # Create an instance of TextExtractor
#     text_extractor = TextExtractor(image_path)

#     # Extract text from the image
#     text_data = text_extractor.extract_text()

#     # Clean the text
#     cleaned_data = TextExtractor.clean_text(text_data)

#     # Now you can use the cleaned text or perform further processing
#     for cleaned_text in cleaned_data:
#         print(f"Cleaned Text: {cleaned_text}")

#     lang_chain_setup = LangChainSetup()
#     response_created = lang_chain_setup.setup_chain(cleaned_text)

#     # Display the final response
#     print(response_created)



# Your existing code...

def process_image_and_get_response(image_path):
    # Create an instance of TextExtractor
    text_extractor = TextExtractor(image_path)

    # Extract text from the image
    text_data = text_extractor.extract_text()

    # Clean the text
    cleaned_data = TextExtractor.clean_text(text_data)

    # Now you can use the cleaned text or perform further processing
    for cleaned_text in cleaned_data:
        print(f"Cleaned Text: {cleaned_text}")

    lang_chain_setup = LangChainSetup()
    response_created = lang_chain_setup.setup_chain(cleaned_text)

    # Return the response_created variable
    return response_created



if __name__ == "__main__":

    image_path = 'dataset/Prescription_12.png'
    response_created = process_image_and_get_response(image_path)

    # Display the final response
    print(response_created)





