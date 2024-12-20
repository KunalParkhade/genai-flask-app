from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )
client = APIClient(credentials)

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

model = ModelInference(
    model_id='mistralai/mistral-large',
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

text = """
<s>[INST]
You are a highly specialized assistant. You only provide answers based on facts and logic. [/INST]

[INST] What is the capital of France? [/INST] </s>
"""

print(model.generate(text)['results'][0]['generated_text'])