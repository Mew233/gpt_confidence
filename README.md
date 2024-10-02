# GPT Confidence - eipm

The goal is to calcualte confidence scores for eipm-gpt4o. This is achieved by calculating the confidence level before and after the model's response. This provides a clear measure of the model's input, helping the physician make better decisions.

This interface show probabilities ('confidence') in the model's response.

It uses the logprobs feature of the OpenAI API and underlines each token in the response. Brighter red means less certainty.
