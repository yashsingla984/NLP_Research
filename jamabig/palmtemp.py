import google.generativeai as palm

# Set your API key
palm.configure(api_key="AIzaSyAILWnvzeDIDPrOuzMahvRPUG7RTniujv8")

#prompt = "What is the meaning of life?"
# models = [m for m in palm.list_models()]
# print(models)
# model = models[0].name
# print(model)
# prompt="Where is Paris ?"
model_id="models/text-bison-001"


prompt="In an ever-evolving global landscape influenced by technological, environmental, and socio-political shifts, how can societies worldwide ensure sustainable economic growth while addressing pressing challenges such as climate change, income inequality, and technological disruption? Explore the intricate interplay between economic development, environmental conservation, and social equity. Assess the efficacy of current global governance structures in fostering inclusive growth and mitigating disparities. Delve into the role of innovation, education, and ethical leadership in shaping a sustainable future. Analyze case studies from diverse regions and industries, evaluating successful models that harmonize economic progress with ecological resilience and social justice. Propose actionable strategies that balance economic prosperity with environmental stewardship and social well-being, promoting a sustainable trajectory for future generations"
#prompt='''Title: Coin-Shaped Opacities in the Stomach Case is: A 50-year-old man with end-stage kidney disease receiving hemodialysis was admitted to the hospital for treatment of calciphylaxis and foot cellulitis. His home medications included sevelamer and hydrocodone-acetaminophen (10 mg/325 mg) every 8 hours as needed, which was increased to every 4 hours as needed in the hospital. Please explain the case '''
#prompt="Title: Coin-Shaped Opacities in the Stomach. Case is: A 50-year-old man with end-stage kidney disease receiving hemodialysis was admitted to the hospital for treatment of calciphylaxis and foot cellulitis. His home medications included sevelamer and hydrocodone-acetaminophen (10 mg/325 mg) every 8 hours as needed, which was increased to every 4 hours as needed in the hospital. "
#Hydromorphone (0.5 mg intravenously as needed) was added for breakthrough pain. He was prescribed chewable lanthanum tablets (500 mg 3 times daily) for treatment of a blood phosphate level of 8.1 mg/dL (reference, 2.5-4.5 mg/dL).On hospital day 7, the patient developed intermittent apneic episodes, during which his oxygen saturation was 80% on room air; heart rate, 86/min; and blood pressure, 106/45 mm Hg. Physical examination revealed bilateral rhonchi and responsiveness to verbal commands only with deep painful stimulus. A chest radiograph showed 4 radio-opaque coin-shaped opacities in the stomach (Figure). After administration of oxygen at 2 L/min by nasal cannula and a naloxone infusion, his oxygen saturation increased to 98% and his mental status improved. The patient reported no foreign body ingestion.Chest radiograph showing 4 rounded objects in patient’s stomach.What Would You Do Next? A: Administer activated charcoal , B: Arrange endoscopy , C: Perform gastric lavage , D: Provide supportive care. Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only
#prompt="What type of prompt you take? I you are able to take any prmpt then why You are not able to answer"
#prompt="the study of  Medical science is the study of the human body and its functions, as well as the prevention, diagnosis, and treatment of diseases.Is it correct ?"
prompt='''Title: Coin-Shaped Opacities in the Stomach Case is: A 50-year-old man with end-stage kidney disease receiving hemodialysis was admitted to the hospital for treatment of calciphylaxis and foot cellulitis. His home medications included sevelamer and hydrocodone-acetaminophen (10 mg/325 mg) every 8 hours as needed, which was increased to every 4 hours as needed in the hospital. Hydromorphone (0.5 mg intravenously as needed) was added for breakthrough pain. He was prescribed chewable lanthanum tablets (500 mg 3 times daily) for treatment of a blood phosphate level of 8.1 mg/dL (reference, 2.5-4.5 mg/dL).On hospital day 7, the patient developed intermittent apneic episodes, during which his oxygen saturation was 80% on room air; heart rate, 86/min; and blood pressure, 106/45 mm Hg. Physical examination revealed bilateral rhonchi and responsiveness to verbal commands only with deep painful stimulus. A chest radiograph showed 4 radio-opaque coin-shaped opacities in the stomach (Figure). After administration of oxygen at 2 L/min by nasal cannula and a naloxone infusion, his oxygen saturation increased to 98% and his mental status improved. The patient reported no foreign body ingestion.Chest radiograph showing 4 rounded objects in patient’s stomach.What Would You Do Next? A: Administer activated charcoal , B: Arrange endoscopy , C: Perform gastric lavage , D: Provide supportive care. Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only and also provide the Explanation:'''
# completion=palm.generate_text(
#     model='models/text-bison-001',
#     prompt=prompt,
#     temperature=0
# )

response=palm.chat(
                        model='models/chat-bison-001',
                        messages=prompt,
                        temperature=0.8,
                        context="Speak like a clinician"
                    )
output = response.candidates[0]['content'] 
#print(response)
# print(completion.result)
print(output)
