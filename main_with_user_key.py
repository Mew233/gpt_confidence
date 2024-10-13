from html import escape
import math
from openai import AzureOpenAI
import streamlit as st
from openai import OpenAI, Stream
from prompts.prompt_design import Prompt
from dotenv import load_dotenv
import os
import re

load_dotenv()

# st.set_page_config(
#     menu_items={
#         "Report a bug": "https://bugs.openai.com/browse",
#     }
# )

with st.sidebar.expander("About ", expanded=True):
    st.markdown(
        """
The goal is to calcualte :rainbow[confidence scores] for eipm-gpt4o. This is achieved by calculating the confidence level :blue[before] and :blue[after] the model's response. This provides a clear measure of the model's input, helping the physician make better decisions.
    
This interface show probabilities ('confidence') in the model's response.
    
It uses the <a href="https://cookbook.openai.com/examples/using_logprobs" target="blank">logprobs</a> feature of the OpenAI API and underlines each token in the response. Brighter red means less certainty.
    
Hover over a token to see the exact confidence and the top 5 other candidates.
    
    
    """,
        unsafe_allow_html=True,
    )


# open_ai_key = st.sidebar.text_input(
#     label="OpenAI API key",
#     placeholder="sk-...",
#     value=os.getenv("OPENAI_API_KEY","sk-..."),
# )

open_ai_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="sk-...",
    value=os.getenv("OPENAI_API_KEY","sk-"),
    type='password',
)
# Add css to hide item with title "Show password text"
st.markdown(
    """
<style>
    [title="Show password text"] {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

def get_client():
    return OpenAI(api_key=st.secrets.OPENAI_API_KEY)


if not open_ai_key:
    st.markdown("👈 To get started, enter your OpenAI API key in the side panel")
    st.caption(
        "(You really shouldn't go pasting your API key into websites that you don't know, but this is an exception because I'm a very trustworthy person.)"
    )
    st.stop()

client = OpenAI(api_key=open_ai_key)

models = ["gpt-4o", "gpt-4o-mini","gpt-4"]
model_name = st.sidebar.radio("Pick a model", options=models, index=0)

prompt_obj = Prompt()
prompt_preamble = prompt_obj.get_prompt_preamble(model='gpt-4', inference_type='adv')

# Initialization
# if 'name' not in st.session_state:
#     st.session_state['name'] = 'Symptoms'

# values = ['<select>',3, 5, 10, 15, 20, 30]
# default_ix = values.index(30)

option = st.sidebar.selectbox(
    "Inference subtype",
    prompt_obj.adv_criteria_to_prompt_mapping.keys(),index=10,
)

reset_button_key = "reset_button"
reset_button = st.sidebar.button("reset",key=reset_button_key)
if reset_button:
    st.session_state.messages = []
    st.session_state['name'] = 'Symptoms'

# a = st.text_area("Patient data example ",
#     value = """I had the pleasure of seeing this patient in consultation regarding the  treatment of her locally recurrent breast cancer on January 09.  As you  know, she is a 73-year-old woman who initially had a right breast mass  removed in February 1994 with an axillary lymph node dissection.  That  surgical procedure revealed a 1 cm, grade 1, infiltrating ductal  carcinoma with clear surgical margins and 21 axillary lymph nodes were  negative for metastatic carcinoma.  S-phase was low at 3.6%.  The tumor  was found to be estrogen-receptor positive and progesterone-receptor  negative.  She received interstitial radiation, which comprised 4500  centigray over a 3 cm diameter.  She received tamoxifen from 1994 to  1996 and stopped due to concern regarding side effects.  At that time,  she was seeing Dr. ***** *****.  In January 1996, a node was  palpated in the left supraclavicular area.  A fine-needle aspiration was  performed and was unremarkable.  A dense area at the 12 o'clock position  in her right breast was tender in 2006 and was felt to be postsurgical  scarring.  This was noted to increase over time in size and density.  Initially, the workup included a PET scan in November 2005
#  that  revealed a 5 x 7 cm area of density consistent with inflammation.  It is  not clear whether she had a CT-guided biopsy at that time or not.  In  February 2008, she noted discomfort in the right anterior chest wall and  again thought that this area might be slightly larger.  She also thought  that she had a new right breast mass.  Workup of that breast mass was  unremarkable.  However, a breast MRI was performed on March 24 that  revealed a bulky irregular mass in the right 8 o'clock posterior breast,  which measured 1.9 x 1.5 cm with heterogenous enhancement.  Concordant  with an area of metabolic uptake on PET/CT scan, there was a second mass  abutting the pectoralis muscle with similar enhancing characteristics  measuring 1.7 x 1.2 cm.  A third nodule was seen along the right lateral  breast measuring 0.6 x 0.3 cm.  The left breast was unremarkable.  The  PET/CT scan had been performed on 02/22/2008, and was compared to a  PET/CT scan in December 2005.  This revealed a 2 cm right axillary  lymph node and a 1.4 cm hypermetabolic soft tissue nodule with an SUV of  4.1 in the right mid anterior chest wall deep in the subcutaneous fat.  A second right axillary lymph node was also noted, which was  hypermetabolic.  Subsequently, Dr. ***** performed a fine-needle  aspiration of the upper medial area, which was positive for carcinoma,  and a core biopsy of the right breast mass, which revealed a reactive  lymph node.
#     """)
# a = """I had the pleasure of seeing this patient in consultation regarding the  treatment of her locally recurrent breast cancer on January 09.  As you  know, she is a 73-year-old woman who initially had a right breast mass  removed in February 1994 with an axillary lymph node dissection.  That  surgical procedure revealed a 1 cm, grade 1, infiltrating ductal  carcinoma with clear surgical margins and 21 axillary lymph nodes were  negative for metastatic carcinoma.  S-phase was low at 3.6%.  The tumor  was found to be estrogen-receptor positive and progesterone-receptor  negative.  She received interstitial radiation, which comprised 4500  centigray over a 3 cm diameter.  She received tamoxifen from 1994 to  1996 and stopped due to concern regarding side effects.  At that time,  she was seeing Dr. ***** *****.  In January 1996, a node was  palpated in the left supraclavicular area.  A fine-needle aspiration was  performed and was unremarkable.  A dense area at the 12 o'clock position  in her right breast was tender in 2006 and was felt to be postsurgical  scarring.  This was noted to increase over time in size and density.  Initially, the workup included a PET scan in November 2005
#      that  revealed a 5 x 7 cm area of density consistent with inflammation.  It is  not clear whether she had a CT-guided biopsy at that time or not.  In  February 2008, she noted discomfort in the right anterior chest wall and  again thought that this area might be slightly larger.  She also thought  that she had a new right breast mass.  Workup of that breast mass was  unremarkable.  However, a breast MRI was performed on March 24 that  revealed a bulky irregular mass in the right 8 o'clock posterior breast,  which measured 1.9 x 1.5 cm with heterogenous enhancement.  Concordant  with an area of metabolic uptake on PET/CT scan, there was a second mass  abutting the pectoralis muscle with similar enhancing characteristics  measuring 1.7 x 1.2 cm.  A third nodule was seen along the right lateral  breast measuring 0.6 x 0.3 cm.  The left breast was unremarkable.  The  PET/CT scan had been performed on 02/22/2008, and was compared to a  PET/CT scan in December 2005.  This revealed a 2 cm right axillary  lymph node and a 1.4 cm hypermetabolic soft tissue nodule with an SUV of  4.1 in the right mid anterior chest wall deep in the subcutaneous fat.  A second right axillary lymph node was also noted, which was  hypermetabolic.  Subsequently, Dr. ***** performed a fine-needle  aspiration of the upper medial area, which was positive for carcinoma,  and a core biopsy of the right breast mass, which revealed a reactive  lymph node.
#      """

a = """
History of Present Illness 
Glioblastoma: WHO grade IV, IDH wild-type by IHC, ATRX preserved, p53 scattered, Ki67 focally up to 
20%, MGMT pending 
Oncomine: PTEN, FGFR3-TACC3 fusion, EGFR amplification 
Foundation not done 
71-year-old Cantonese-speaking woman with right frontal GBM (IDH wild-type, MGMT not yet resulted) s/p 
resection by Dr. Schwartz on 12/18/19, s/p RT/TMZ (40 Gy in 15 fractions, 1/15/20-2/5/20) presenting for 
follow-up. 
Regarding her neuro-oncologic history, patient was had general memory loss for approximately 1 year 
attributed to aging. In the month prior to presentation, memory impairment dramatically worsened to the 
point that patient was unable to find her way home after leaving the house. She was also noted to be more 
somnolent, falling asleep frequently including during meals. She saw an outside neurologist who 
performed an MRI brain which revealed a left frontal mass lesion with surrounding edema. He was 
admitted to WCMC on 12/13/19. MRI brain with and without contrast performed 12/14/19 revealed an 
enhancing right frontal mass crossing the corpus callosum with surrounding vasogenic edema as well as 
non-enhancing T2/FLAIR hyperintense lesions in the right precentral gyrus, left frontal region, and right 
thalamus. She underwent right frontal craniotomy and resection of the right frontal mass on 12/18/19 by 
Dr. Schwartz; pathology revealed GBM (IDH wild-type by IHC, MGMT promoter methylation status 
pending). Post-operatively, she required insulin intermittently for steroid-induced hyperglycemia. PT/OT 
recommended discharge to SAR, but family requested that patient be discharged home with 24-hour 
supervision. 
Patient underwent hypofractionated course of RT/TMZ (40 Gy in 15 fractions) from 1/15/20-2/5/20 with Dr. 
Knisely. The decision was made not to pursue Foundation Medicine testing on tumor tissue given patient's 
advanced age, neurologic deficits, and frailty. Daughter reported that patient had difficulty standing from a 
seated position with moon facies on 1/31/20; she was instructed to stop dexamethasone. However, at 
Radiation Oncology appointment on 2/5/20, she was still taking dexamethasone 1 mg daily; she was 
instructed to take 1 mg qod for 1 week then stop dexamethasone. 
Today, patient presents with her daughter for follow-up. Daughter states that patient had fever and cough 
1-2 weeks ago and that she went to her family doctor; symptoms are now improved. She has had no 
recent travel outside the country nor contact with anyone who has traveled outside the country recently. 
Daughter feels that patient's legs are weaker and that she has had hair loss following RT. She is no longer 
taking dexamethasone; her face and legs were swollen on dexamethasone, but swelling has now improved. Patient reports pain in her left leg. Cognitively, daughter feels that patient's memory/thicking are 
possibly slightly worse; when she wakes up, she is more confused, but confusion improves throughout the 
day. She used to be able to write her name but now is unable to do so. 
Assessment & Plan 
71-year-old Cantonese-speaking woman with right frontal GBM (IDH wild-type, MGMT not yet resulted) s/p 
resection by Dr. Schwartz on 12/18/19, s/p RT/TMZ (40 Gy in 15 fractions, 1/15/20-2/5/20) presenting for 
follow-up. Neurologic exam at this time remains notable for disorientation to time, difficulty following 
complex commands, and marked bradyphrenia with psychomotor slowing as well as shuffling gait 
necessitating a rolling walker. MRI brain performed today approximately 1 month after the completion of 
RT reveals new nodular enhancement in the left frontal lobe distant from the initial resection cavity as well 
as new nodular enhancement along the margins of the resection site extending along the right lateral 
ventricle and involving the corpus callosum concerning for progression of disease. At this time, given the 
rapidity of her disease progression, and given her frailty and significant cognitive/functional impairment, 
further treatment carries a significant risk of adversely affecting her quality of life while offering little benefit. 
We discussed that she has multifocal disease which is not amenable to surgery or further RT, and that 
there are chemotherapies that can be trialed, but that these treatments are unlikely to be significantly
effective or improve her quality of life while posing a substantial risk of side effects given her neurologic 
deficits and frailty. Given that the risks of further treatment outweigh any potential benefit, would 
recommend focusing care on comfort and pursuing home hospice. We discussed that, at some point, her 
tumor would grow further and end her life; we discussed that hospice services could help treat any 
symptoms she may have and help maintain her comfort and quality of life. Patient's daughter expressed 
understanding and stated that, rather than having our office make referrals to hospice now, she would like 
to set up hospice through the patient's family doctor. We encouraged patient's daughter to call us, as we 
are happy to assist in any way possible. 
Plan: 
1) Right frontal/multifocal GBM: rapid progression of disease 
-Recommend pursuing hospice at this time as above. Daughter would like to set up hospice through 
patient's family doctor and will call if they require any assistance 
-No further MRls 
2) Steroids: Off steroids as of 2/2020, no indication to re-start 
3) AEDs: No history of seizures. 
-Continue Keppra 500mg bid for seizure prevention 
4) Cognitive impairment, frailty: Recommend home hospice as above
"""

if  st.button('Patient data example'):
    st.code(a, language='None')
    st.write(f"{len(a)} characters.")
    # pyperclip.copy(a)
    # st.success('Text copied successfully!')







if "messages" not in st.session_state:
    st.session_state.messages = []


def stream_to_html(stream: Stream):
    html = ""
    with st.empty():
        for chunk in stream:
            choice = chunk.choices[0]
            text = chunk.choices[0].delta.content

            if not text:
                continue

            text = escape(text)
            content = choice.logprobs.content[0]
            if "\n" in content.token:
                # Tokens can be :\n\n, etc, ideally these would be split into returns and content
                html += content.token.replace("\n", "<br>")
            else:
                prob = math.exp(content.logprob)
                underline = f"3px solid rgba(255, 0, 0, {1-prob ** 1.6:.0%})"
                options = [
                    f"{escape(x.token)} ({math.exp(x.logprob):.2%})"
                    for x in content.top_logprobs
                ]
                tooltip = "\n".join(options)
                span = f"<span title='{tooltip}' style='border-bottom: {underline}'>{text}</span>"
                html += span

            st.html(html)

    return html


avatars = dict(
    assistant='<svg viewBox="-4 -4 50 50" fill="none" xmlns="http://www.w3.org/2000/svg" role="img"><path d="M37.5324 16.8707C37.9808 15.5241 38.1363 14.0974 37.9886 12.6859C37.8409 11.2744 37.3934 9.91076 36.676 8.68622C35.6126 6.83404 33.9882 5.3676 32.0373 4.4985C30.0864 3.62941 27.9098 3.40259 25.8215 3.85078C24.8796 2.7893 23.7219 1.94125 22.4257 1.36341C21.1295 0.785575 19.7249 0.491269 18.3058 0.500197C16.1708 0.495044 14.0893 1.16803 12.3614 2.42214C10.6335 3.67624 9.34853 5.44666 8.6917 7.47815C7.30085 7.76286 5.98686 8.3414 4.8377 9.17505C3.68854 10.0087 2.73073 11.0782 2.02839 12.312C0.956464 14.1591 0.498905 16.2988 0.721698 18.4228C0.944492 20.5467 1.83612 22.5449 3.268 24.1293C2.81966 25.4759 2.66413 26.9026 2.81182 28.3141C2.95951 29.7256 3.40701 31.0892 4.12437 32.3138C5.18791 34.1659 6.8123 35.6322 8.76321 36.5013C10.7141 37.3704 12.8907 37.5973 14.9789 37.1492C15.9208 38.2107 17.0786 39.0587 18.3747 39.6366C19.6709 40.2144 21.0755 40.5087 22.4946 40.4998C24.6307 40.5054 26.7133 39.8321 28.4418 38.5772C30.1704 37.3223 31.4556 35.5506 32.1119 33.5179C33.5027 33.2332 34.8167 32.6547 35.9659 31.821C37.115 30.9874 38.0728 29.9178 38.7752 28.684C39.8458 26.8371 40.3023 24.6979 40.0789 22.5748C39.8556 20.4517 38.9639 18.4544 37.5324 16.8707ZM22.4978 37.8849C20.7443 37.8874 19.0459 37.2733 17.6994 36.1501C17.7601 36.117 17.8666 36.0586 17.936 36.0161L25.9004 31.4156C26.1003 31.3019 26.2663 31.137 26.3813 30.9378C26.4964 30.7386 26.5563 30.5124 26.5549 30.2825V19.0542L29.9213 20.998C29.9389 21.0068 29.9541 21.0198 29.9656 21.0359C29.977 21.052 29.9842 21.0707 29.9867 21.0902V30.3889C29.9842 32.375 29.1946 34.2791 27.7909 35.6841C26.3872 37.0892 24.4838 37.8806 22.4978 37.8849ZM6.39227 31.0064C5.51397 29.4888 5.19742 27.7107 5.49804 25.9832C5.55718 26.0187 5.66048 26.0818 5.73461 26.1244L13.699 30.7248C13.8975 30.8408 14.1233 30.902 14.3532 30.902C14.583 30.902 14.8088 30.8408 15.0073 30.7248L24.731 25.1103V28.9979C24.7321 29.0177 24.7283 29.0376 24.7199 29.0556C24.7115 29.0736 24.6988 29.0893 24.6829 29.1012L16.6317 33.7497C14.9096 34.7416 12.8643 35.0097 10.9447 34.4954C9.02506 33.9811 7.38785 32.7263 6.39227 31.0064ZM4.29707 13.6194C5.17156 12.0998 6.55279 10.9364 8.19885 10.3327C8.19885 10.4013 8.19491 10.5228 8.19491 10.6071V19.808C8.19351 20.0378 8.25334 20.2638 8.36823 20.4629C8.48312 20.6619 8.64893 20.8267 8.84863 20.9404L18.5723 26.5542L15.206 28.4979C15.1894 28.5089 15.1703 28.5155 15.1505 28.5173C15.1307 28.5191 15.1107 28.516 15.0924 28.5082L7.04046 23.8557C5.32135 22.8601 4.06716 21.2235 3.55289 19.3046C3.03862 17.3858 3.30624 15.3413 4.29707 13.6194ZM31.955 20.0556L22.2312 14.4411L25.5976 12.4981C25.6142 12.4872 25.6333 12.4805 25.6531 12.4787C25.6729 12.4769 25.6928 12.4801 25.7111 12.4879L33.7631 17.1364C34.9967 17.849 36.0017 18.8982 36.6606 20.1613C37.3194 21.4244 37.6047 22.849 37.4832 24.2684C37.3617 25.6878 36.8382 27.0432 35.9743 28.1759C35.1103 29.3086 33.9415 30.1717 32.6047 30.6641C32.6047 30.5947 32.6047 30.4733 32.6047 30.3889V21.188C32.6066 20.9586 32.5474 20.7328 32.4332 20.5338C32.319 20.3348 32.154 20.1698 31.955 20.0556ZM35.3055 15.0128C35.2464 14.9765 35.1431 14.9142 35.069 14.8717L27.1045 10.2712C26.906 10.1554 26.6803 10.0943 26.4504 10.0943C26.2206 10.0943 25.9948 10.1554 25.7963 10.2712L16.0726 15.8858V11.9982C16.0715 11.9783 16.0753 11.9585 16.0837 11.9405C16.0921 11.9225 16.1048 11.9068 16.1207 11.8949L24.1719 7.25025C25.4053 6.53903 26.8158 6.19376 28.2383 6.25482C29.6608 6.31589 31.0364 6.78077 32.2044 7.59508C33.3723 8.40939 34.2842 9.53945 34.8334 10.8531C35.3826 12.1667 35.5464 13.6095 35.3055 15.0128ZM14.2424 21.9419L10.8752 19.9981C10.8576 19.9893 10.8423 19.9763 10.8309 19.9602C10.8195 19.9441 10.8122 19.9254 10.8098 19.9058V10.6071C10.8107 9.18295 11.2173 7.78848 11.9819 6.58696C12.7466 5.38544 13.8377 4.42659 15.1275 3.82264C16.4173 3.21869 17.8524 2.99464 19.2649 3.1767C20.6775 3.35876 22.0089 3.93941 23.1034 4.85067C23.0427 4.88379 22.937 4.94215 22.8668 4.98473L14.9024 9.58517C14.7025 9.69878 14.5366 9.86356 14.4215 10.0626C14.3065 10.2616 14.2466 10.4877 14.2479 10.7175L14.2424 21.9419ZM16.071 17.9991L20.4018 15.4978L24.7325 17.9975V22.9985L20.4018 25.4983L16.071 22.9985V17.9991Z" fill="white"></path></svg>',
    user=""":material/person:""",
)

for message in st.session_state.messages:
    role = message["role"]
    text = message["content"]
    with st.chat_message(role, avatar=avatars.get(role)):
        if role == "user":
            st.markdown(text, unsafe_allow_html=True)
        else:
            st.html(text)



# _plc = " I had the pleasure of seeing this patient in consultation regarding the  treatment of her locally recurrent breast cancer on January 09.  As you  know, she is a 73-year-old woman who initially had a right breast mass  removed in February 1994 with an axillary lymph node dissection.  That  surgical procedure revealed a 1 cm, grade 1, infiltrating ductal  carcinoma with clear surgical margins and 21 axillary lymph nodes were  negative for metastatic carcinoma.  S-phase was low at 3.6%.  The tumor  was found to be estrogen-receptor positive and progesterone-receptor  negative.  She received interstitial radiation, which comprised 4500  centigray over a 3 cm diameter.  She received tamoxifen from 1994 to  1996 and stopped due to concern regarding side effects.  At that time,  she was seeing Dr. ***** *****.  In January 1996, a node was  palpated in the left supraclavicular area.  A fine-needle aspiration was  performed and was unremarkable.  A dense area at the 12 o'clock position  in her right breast was tender in 2006 and was felt to be postsurgical  scarring.  This was noted to increase over time in size and density.  Initially, the workup included a PET scan in November 2005  that  revealed a 5 x 7 cm area of density consistent with inflammation.  It is  not clear whether she had a CT-guided biopsy at that time or not.  In  February 2008, she noted discomfort in the right anterior chest wall and  again thought that this area might be slightly larger.  She also thought  that she had a new right breast mass.  Workup of that breast mass was  unremarkable.  However, a breast MRI was performed on March 24 that  revealed a bulky irregular mass in the right 8 o'clock posterior breast,  which measured 1.9 x 1.5 cm with heterogenous enhancement.  Concordant  with an area of metabolic uptake on PET/CT scan, there was a second mass  abutting the pectoralis muscle with similar enhancing characteristics  measuring 1.7 x 1.2 cm.  A third nodule was seen along the right lateral  breast measuring 0.6 x 0.3 cm.  The left breast was unremarkable.  The  PET/CT scan had been performed on 02/22/2008, and was compared to a  PET/CT scan in December 2005.  This revealed a 2 cm right axillary  lymph node and a 1.4 cm hypermetabolic soft tissue nodule with an SUV of  4.1 in the right mid anterior chest wall deep in the subcutaneous fat.  A second right axillary lymph node was also noted, which was  hypermetabolic.  Subsequently, Dr. ***** performed a fine-needle  aspiration of the upper medial area, which was positive for carcinoma,  and a core biopsy of the right breast mass, which revealed a reactive  lymph node. "
if cur_text_ := st.chat_input("Input the patient's note"):
    template, _2ndtemplate, subprompt = prompt_obj.get_prompt(inference_type='sdoh_entity', inference_subtype=option) #option
    prompt = prompt_preamble + template.format(cur_text_,subprompt)

    _2nd_prompt = prompt_preamble + _2ndtemplate.format(cur_text_,subprompt)

    st.session_state.messages.append(dict(role="user", content=prompt))
    with st.chat_message("user", avatar=avatars.get("user")):
        st.markdown(template.format(cur_text_,subprompt))
        st.session_state.messages.append(dict(role="assistant", content=prompt))

    with st.chat_message("assistant", avatar=avatars.get("assistant")):
        stream = client.chat.completions.create(
            model=model_name,
            messages=st.session_state.messages,
            logprobs=True,
            top_logprobs=5,
            stream=True,
            temperature = 0
        )
        text = stream_to_html(stream)

        st.session_state.messages.append(dict(role="assistant", content=text))

        final_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": _2nd_prompt
            }])

        _text = st.markdown("```python\n" + final_response.choices[0].message.content + "\n```")
        st.session_state.messages.append(dict(role="assistant", content=_text))

        # Split the text into individual data entries based on newlines
        # entries = final_response.choices[0].message.content.strip().split('\n')

        # Regular expression to match everything after --FINAL ANSWER--
        pattern = r'----FINAL ANSWER----\s*(.*)'
        input_text = final_response.choices[0].message.content
        # Extracting the text after --FINAL ANSWER--
        match = re.search(pattern, input_text, re.DOTALL)

        # Check if there is a match and print the extracted text
        if match:
            extracted_text = match.group(1).strip()  # Strip any extra whitespace/newlines
            print(extracted_text)

            entries = extracted_text.split('\n')

            # Iterate and print each entry
            for index, entry in enumerate(entries, start=1):
                print(f"Data {index}: {entry}")
                # if index == 1 or index == len(entries):
                #     continue

                boolean = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Based on your previous prompt and result extracted. "
                                       "Answer If your result are mataching the records in the  clinical note; you are not answering the actual question."
                                       "Respond with just one word, the Boolean yes or no. You must output the word 'yes', or the word 'no', nothing else."
                        },
                        {
                            "role": "assistant",
                            "content": f"Your previous prompt:{_2nd_prompt}"
                        },
                        {
                            "role": "assistant",
                            "content": f"Result extracted: {entry}"
                        }],
                    logprobs=True,
                    top_logprobs=5,
                    stream=True,
                    temperature = 0)

                _boolean = stream_to_html(boolean)
                st.session_state.messages.append(dict(role="assistant", content=_boolean))
                st.markdown("```python\n" +entry+"\n```")



if not st.session_state.messages:
    st.write("Have  patient data down the bottom 👇")
