import google.generativeai as palm

# Set your API key
palm.configure(api_key="AIzaSyCV8F7nV8C3OeSqEfpi1O1_Bv5p42QR1Uw")

# Define your prompt
prompt = "What is the meaning of life?"

prompt="Where is Paris ?"

# models = [model for model in palm.list_models()]

# for model in models:
#   print(model.name)


# Generate text using PaLM
# completion = palm.generate_text(
#     model="models/chat-bison-001",
#     prompt=prompt,
#     temperature=0.7,
#     max_output_tokens=200,
# )

# # Print the generated text
# print(completion.result)


model_id="models/text-bison-001"
# prompt='''write a cover letter for a data science job applicaton. 
# Summarize it to two paragraphs of 50 words each. '''

# prompt='''Title: Coin-Shaped Opacities in the Stomach Case is: A 50-year-old man with end-stage kidney disease receiving hemodialysis was admitted to the hospital for treatment of calciphylaxis and foot cellulitis. His home medications included sevelamer and hydrocodone-acetaminophen (10 mg/325 mg) every 8 hours as needed, which was increased to every 4 hours as needed in the hospital. Hydromorphone (0.5 mg intravenously as needed) was added for breakthrough pain. He was prescribed chewable lanthanum tablets (500 mg 3 times daily) for treatment of a blood phosphate level of 8.1 mg/dL (reference, 2.5-4.5 mg/dL).On hospital day 7, the patient developed intermittent apneic episodes, during which his oxygen saturation was 80% on room air; heart rate, 86/min; and blood pressure, 106/45 mm Hg. Physical examination revealed bilateral rhonchi and responsiveness to verbal commands only with deep painful stimulus. A chest radiograph showed 4 radio-opaque coin-shaped opacities in the stomach (Figure). After administration of oxygen at 2 L/min by nasal cannula and a naloxone infusion, his oxygen saturation increased to 98% and his mental status improved. The patient reported no foreign body ingestion.Chest radiograph showing 4 rounded objects in patient’s stomach.What Would You Do Next? A: Administer activated charcoal , B: Arrange endoscopy , C: Perform gastric lavage , D: Provide supportive care. Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only'''
#prompt="Title: Coin-Shaped Opacities in the Stomach Case is: A 50-year-old man with end-stage kidney disease receiving hemodialysis was admitted to the hospital for treatment of calciphylaxis and foot cellulitis. His home medications included sevelamer and hydrocodone-acetaminophen (10 mg/325 mg) every 8 hours as needed, which was increased to every 4 hours as needed in the hospital. Hydromorphone (0.5 mg intravenously as needed) was added for breakthrough pain. He was prescribed chewable lanthanum tablets (500 mg 3 times daily) for treatment of a blood phosphate level of 8.1 mg/dL (reference, 2.5-4.5 mg/dL).On hospital day 7, the patient developed intermittent apneic episodes, during which his oxygen saturation was 80% on room air; heart rate, 86/min; and blood pressure, 106/45 mm Hg. Physical examination revealed bilateral rhonchi and responsiveness to verbal commands only with deep painful stimulus. A chest radiograph showed 4 radio-opaque coin-shaped opacities in the stomach (Figure). After administration of oxygen at 2 L/min by nasal cannula and a naloxone infusion, his oxygen saturation increased to 98% and his mental status improved. The patient reported no foreign body ingestion.Chest radiograph showing 4 rounded objects in patient’s stomach.What Would You Do Next? A: Administer activated charcoal , B: Arrange endoscopy , C: Perform gastric lavage , D: Provide supportive care. Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
completion=palm.generate_text(
    model=model_id,
    prompt=prompt,
    temperature=0,
    max_output_tokens=1800,
)
print(completion)
print(completion.result)
