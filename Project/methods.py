import os
from flask import jsonify
from langdetect import detect
from langchain.prompts import PromptTemplate 
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
import os
from langchain_core.messages import HumanMessage
import os
import json
import sqlite3
import pandas as pd
import wikipedia
import os
from langchain_community.utilities import SerpAPIWrapper
from serpapi import GoogleSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
import torch
import faiss
import json
from transformers import CLIPModel, CLIPProcessor
import speech_recognition as sr
from langchain.chains import RetrievalQA

from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
 
os.environ["SERPAPI_API_KEY"] = 
os.environ["huggingface_token"] = 
os.environ["OPENAI_API_KEY"] = 

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
OPENAI_API_KEY=

 
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
wiki = WikipediaAPIWrapper(lang="ar")  # أو "en" حسب اللغة


google_search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

  


#----------------------------------------------------------------------


# 1. تحميل JSON كقائمة من القواميس

json_path = "data/landmarks_list2.json"   # غيّر إلى مسار ملفك
if not os.path.exists(json_path):
    raise FileNotFoundError(f"لم أجد الملف: {json_path}")

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)              # data: List[Dict]

# 2. إزالة التكرارات (مثال على أساس العنوان)
seen = set()
unique = []
for item in data:
    title = item.get("title")
    if title and title not in seen:
        seen.add(title)
        unique.append(item)
print(f"✅ إزالة التكرارات: بقي {len(unique)} عنصرًا")

# 3. تحويل إلى DataFrame
df = pd.DataFrame(unique)
df["id"] = df.index.astype(str)     # مفتاح فريد

# 4. بناء جدول metadata في SQLite (اختياري ولكن مفيد للبحث الوصفي)
conn = sqlite3.connect("landmarks2.db")
df[["id", "title", "url"]].to_sql(
    "landmarks_meta", conn, if_exists="replace", index=False
)
conn.close()
print("✅ تم إنشاء جدول metadata في landmarks2.db")

# 5. إعداد Embeddings + Vector Store (FAISS)
texts     = df["description"].tolist()   # أو حقل آخر يحتوي النصوص المراد تضمينها
metadatas = df[["id", "title", "url"]].to_dict(orient="records")

 
# تحويلها إلى مستندات LangChain
documents = [Document(page_content=item["description"]) for item in data]

# chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# توليد embeddings + إنشاء قاعدة FAISS
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)

# حفظها للاسترجاع لاحقًا
vectorstore.save_local("faiss_landmarks")


embeddings = OpenAIEmbeddings()      # تأكد من ضبط OPENAI_API_KEY
vectordb   = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectordb.save_local("faiss_landmarks")
print("✅ فهرس FAISS جاهز ومحفوظ")

# 6. تحميل الفهرس وتهيئة Retriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# your embeddings
embeddings = OpenAIEmbeddings()

# load the FAISS index you saved earlier, allowing your own pickle
vectordb = FAISS.load_local(
    "faiss_landmarks",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3}) 

 
#----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/labels.json") as f:
    labels = json.load(f)



def detect_language(text):
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except:
        return "en"  # fallback
    
def enhance_query_for_tourism(query: str, lang="ar"):
    if lang == "ar":
        return f"معلم سياحي {query}"
    else:
        return f"tourist attraction {query}"


def get_prompt_by_lang( lang="ar"):
    if lang == "ar":
        prompt  = PromptTemplate(
        input_variables=["context", "landmark"],
        template="""أنت مرشد سياحي ذكي وخبير.

        بناءً على المعلومات التالية، أنشئ فقرة جذابة ومفيدة للسائح عن المعلم السياحي "{landmark}" تتضمن ما يلي:
 (مع تضمين ستايل كود html بدون روابط وأعرض الصورة )
        1. وصف مميز للمعلم السياحي وتاريخه وموقعه وأهميته.
        2. وسائل المواصلات المناسبة للوصول إليه (مثل المترو، أوبر، كريم، أو سيارات الأجرة).
        3. تطبيقات محلية مفيدة للسياح في المنطقة (مثل الخرائط أو الترجمة أو المواصلات).
        4. اقتراح مقطع أو مقطعين من فيديوهات مشهورة على يوتيوب تتحدث عن المعلم أو المدينة.
        5. أسماء 2 إلى 3 شركات سياحية محلية موثوقة يمكنها تنظيم زيارة للمعلم.
        6. إذا توفرت، أعرض الصورة المرسلة في {context}.

        📌 التعليمات:
        - لا تخترع معلومات غير موجودة في السياق.
        - إذا لم تتوفر معلومات كافية، أذكر أن المعلومات محدودة.
        - اجعل النص جذابًا وسهل القراءة، ومناسبًا للسياح.
        - اكتب بأسلوب احترافي ومتحمس.
        """
        )
        return prompt
    else: 
        prompt  = PromptTemplate(
        input_variables=["context", "landmark"],
        template="""You are a smart and experienced tour guide.

        Based on the following information, create an engaging and informative paragraph (with html tags and no links and show img)
        for tourists about the landmark "{landmark}" that includes the following:

        1. A distinctive description of the landmark, its history, location, and significance.
        2. Suitable means of transportation for reaching it (such as the metro, Uber, Careem, or taxis).
        3. Useful local apps for tourists in the area (such as maps, translation, or transportation).
        4. Suggest one or two popular YouTube videos about the landmark or city.
        5. The names of two to three reliable local tour companies that can arrange a visit to the landmark.
        6. If available, display the submitted image in {context}.
        📌 Instructions:
        - Don't invent information that isn't relevant to the context.
        - If there isn't enough information, state that the information is limited.
        - Make the text engaging, easy to read, and tourist-friendly.
        - Write in a professional and enthusiastic style.
        """
        )
        return prompt

def get_prompt_template_by_lang( lang="ar"):
    if lang == "ar":
        prompt  = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        أنت مرشد سياحي ذكي. 
        استخدم المعلومات التالية للإجابة على السؤال بدقة وبأسلوب مختصر:     
         (مع تضمين ستايل كود html بدون روابط  )

       📚 السياق:  
       {context}  

        📌 التعليمات:
        - لا تخترع معلومات غير موجودة في السياق.
        - إذا لم تتوفر معلومات كافية، أذكر أن المعلومات محدودة.
        - اجعل النص جذابًا وسهل القراءة، ومناسبًا للسياح.
        - اكتب بأسلوب احترافي ومتحمس.

       ❓ السؤال:
        {question}

        🧠 الجواب:
 
        """
        )
        return prompt
    else: 
        prompt  = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a smart tour guide.
        Use the following information to answer the question accurately and concisely:  
        (with html tags and no links and show img if it's work)

        📚 Context:
        {context}

        📌 Instructions:
        - Don't invent information not found in the context.
        - If not enough information is available, state that the information is limited.
        - Make the text engaging, easy to read, and tourist-friendly.
        - Write in a professional and enthusiastic style.

        ❓ Question:
        {question}

        🧠 Answer:  
        """
        )
        return prompt

def get_context_by_lang( lang="ar"):
    if lang == "ar":
        context  = """
             1. وصف مميز للمعلم السياحي وتاريخه وموقعه وأهميته.
        2. وسائل المواصلات المناسبة للوصول إليه (مثل المترو، أوبر، كريم، أو سيارات الأجرة).
        3. تطبيقات محلية مفيدة للسياح في المنطقة (مثل الخرائط أو الترجمة أو المواصلات).
        4. اقتراح مقطع أو مقطعين من فيديوهات مشهورة على يوتيوب تتحدث عن المعلم أو المدينة.
        5. أسماء 2 إلى 3 شركات سياحية محلية موثوقة يمكنها تنظيم زيارة للمعلم.
        """
        return context
    else: 
        context="""
       
        1. A distinctive description of the landmark, its history, location, and significance.
        2. Suitable means of transportation for reaching it (such as the metro, Uber, Careem, or taxis).
        3. Useful local apps for tourists in the area (such as maps, translation, or transportation).
        4. Suggest one or two popular YouTube videos about the landmark or city.
        5. The names of two to three reliable local tour companies that can arrange a visit to the landmark.
        """
        
        return context

        # 6. If available, display the submitted image in {context}.

 
def search_tourist_images_on_google(landmark_name):
    params = {
        "engine": "google",
        "q": f"{landmark_name}  ",
        "tbm": "isch",   
        "api_key": os.getenv("SERPAPI_API_KEY")  
    }
 
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get("images_results", [])
    return [img["original"] for img in images[:5]] 


def smart_search(query: str, sentences=5,  lang="ar"):
    
    try: 
        wikipedia.set_lang(lang)
        results = wikipedia.search(query)
        if results:
            title = results[0]
            page = wikipedia.page(title, auto_suggest=False)
            summary = wikipedia.summary(title, sentences=sentences)


            # images = search_tourist_images_on_google(query)
            image_url = next((img for img in page.images if img.lower().endswith(('.jpg', '.png'))), None)
 
            if "wikimedia"  not in image_url:
                image_url = None
 
            return summary + (f"\n\n🖼️ صورة: {image_url}" if image_url else ""), image_url
    except:
        pass
 

    # 3️⃣ أخيرًا: بحث جوجل (SerpAPI)
    try:
        serp_result = google_search.run(query)
        return f"📍 تم جلب المحتوى من Google:\n\n{serp_result}", ""
    except:
        return "❌ لم يتم العثور على معلومات كافية حتى من Google.", ""


def extract_landmark_name_from_text(user_input):
    system_prompt = (
        "Extract the name of the tourist attraction from the user's question. "
        "Just return the name and the name of city if found only, with no extra words."
    )
    response = llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": user_input}
        ])
    ])
    return response.content.strip()
 
# الدالة الأساسية
# Smart Agent => Reg in 
def text_tool_func(query: str) -> str: 
    prompt  = get_prompt_by_lang(detect_language(query))
 
    query = enhance_query_for_tourism(query , detect_language(query))

    landmark = extract_landmark_name_from_text(query)
    context, image_url =  smart_search (landmark , 5 , detect_language(landmark))
    
    if "❌" in context:
        return f"📍 {landmark}\n{context}"
    
    # أضف الصورة في النهاية
    full_context = context + f"\n\n🔗 صورة المعلم:\n{image_url}"
    
    final_prompt = prompt.format(context=full_context, landmark=landmark)
    return llm.invoke(final_prompt).content



  
 
def  rag_tool_func(query: str) -> str: 
    prompt = get_prompt_template_by_lang(detect_language(query)) 
    landmark = extract_landmark_name_from_text(query) 

    sourse = "Source of information from the knowledge base"
    qa_chain = RetrievalQA.from_chain_type( 
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # يستخدم البرومبت مع كل الوثائقprompt
        chain_type_kwargs={"prompt":  prompt },
        return_source_documents=True
    )
 
    try:
        rag_result = qa_chain({"query": query})
        answer = rag_result.get("result", "")
        
        if not answer.strip() or "لا توجد" in answer or "doesn't include" in answer  or "عذرًا" in answer or len(answer) < 30:
            # fallback to Wikipedia
            print("🔍 Falling back to Wikipedia...")
            sourse = "Source of information from Wikipedia"

            Wiki_result =   text_tool_func(query) 

            return  {
            'answer': Wiki_result, 
            "lang": detect_language(query)  ,
            'sourse': sourse , 
    } 
            
         
        return   {
            'answer': answer, 
            "lang": detect_language(query)  ,
            'sourse': sourse , 
    } 
    
        
    except Exception as e:
        return f"❌ Error occurred: {e}" , "ar" , "error"
    
 
def recognize_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs).cpu().numpy().astype('float32')
    _, indices = index.search(img_emb, k=1)
    idx = indices[0][0]
    return labels[idx] if idx < len(labels) else "Unknown Landmark"


def image_tool_func(image_path: str) -> str:
    landmark = recognize_image(image_path)
     
    if landmark == "Unknown Landmark":
        return "❌ لم يتم التعرف على المعلم في الصورة."
    
    return rag_tool_func(landmark) 
 




def audio_tool_func(audio_path: str , lang="en") -> str:
    r = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)

    # تعرف على الكلام
    text = r.recognize_google(audio ,  language= lang )
    landmark = extract_landmark_name_from_text(text) 
    return rag_tool_func(landmark) 
 

text_tool = Tool(
    name="RAGTourGuide",
    func=rag_tool_func,
    description="يتعرف على المعلم من النص ويرجع وصفًا سياحيًا كاملاً.",
    return_direct=True
)

image_tool = Tool(
    name="ImageTourGuide",
    func=image_tool_func,
    description="يتعرف على المعلم من صورة ويرجع وصفًا سياحيًا كاملاً.",
    return_direct=True
)


audio_tool = Tool(
    name="AudioTourGuide",
    func=audio_tool_func,
    description="يتعرف على المعلم من الصوت ويرجع وصفًا سياحيًا كاملاً.",
    return_direct=True
)


agent = initialize_agent(
    tools=[text_tool, image_tool, audio_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


def  multiple_values(result ):
      
    return isinstance(result, (tuple, list, dict)) 


# Start Here =)
def runSmartAgent( user_input ):

    resulte = agent.run( user_input )
     
    if multiple_values (resulte):
        answer = resulte["answer"].replace( "```html", '').replace( "```", '') 
        lang = resulte["lang"]
        sourse = resulte["sourse"]
        
        return answer , lang , sourse
   
    else:
        answer = resulte
        lang ="en"  
        sourse ="Source of information from the knowledge base"
        return answer , lang , sourse
        
     
    
  

    
