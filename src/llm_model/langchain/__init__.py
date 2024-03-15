from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback


class LangChainSetup:
    def __init__(self):
        KEY = "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"    #Secret key  
        self.llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.3)

    def setup_chain(self, cleaned_text):
        Text = cleaned_text
        
        RESPONSE_JSON = {
    "1": {
        'name': 'name of the Medicine',
        'Content Value': 'Dosage value in mg (Milli Gram) or ml (Millilitre)',
        'Dosage': 'Dosage frequency'
    },
    "2": {
        'name': 'name of the Medicine',
        'Content Value': 'Dosage value in mg (Milli Gram) or ml (Millilitre)',
        'Dosage': 'Dosage frequency'
    },
    "3": {
        'name': 'name of the Medicine',
        'Content Value': 'Dosage value in mg (Milli Gram) or ml (Millilitre)',
        'Dosage': 'Dosage frequency (How many times in a day it should take)'
    },
}
        

        TEMPLATE = """
        Text:{text}
        You are an expert of Medicines. Given the above text, it is your job to extract the medicine information from it.
        Make sure you extract all medicine from text, check all medicines Dosage value in mg and Dosage frequency carefully.
        Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
        Ensure to extract all medicine from text.
        ### RESPONSE_JSON
        {response_json}
        """

        Prescription_prompt = PromptTemplate(
            input_variables=["text", "response_json"],
            template=TEMPLATE
        )

        precription_chain = LLMChain(llm=self.llm, prompt=Prescription_prompt, output_key="medicine", verbose=True)

        TEMPLATE2 = """
        You are an expert english grammarian and writer. Given a name of the Medicines.\
        You need to correct any spelling errors and provide the corrected medicines name.
        Medicine Information:
        {medicine}

        Check from an expert English Writer of the above medicine information:
        """

        precrition_evaluation_prompt = PromptTemplate(input_variables=["medicine"], template=TEMPLATE2)

        review_chain = LLMChain(llm=self.llm, prompt=precrition_evaluation_prompt, output_key="review", verbose=True)

        generate_evaluate_chain = SequentialChain(chains=[precription_chain, review_chain],
                                                  input_variables=["text", "response_json"],
                                                  output_variables=["medicine", "review"], verbose=True)
        

        # How to setup Token Usage Tracking in LangChain
        with get_openai_callback() as cb:
            response = generate_evaluate_chain(
                {
                    "text": Text,
                    "response_json": RESPONSE_JSON
                }
            )

        response_created = response.get("medicine")
        return response_created
