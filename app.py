import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from twit import tweeter

import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": "es",  # Código de país para España
        "hl": "es",  # Idioma español
        "autocorrect": True
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")



def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
    will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact
    Do not describe what the webpage is, you are here to get acurate and specific information
    Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
    Here is the text:

    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
content="""Eres un investigador de primera clase, capaz de realizar investigaciones detalladas sobre cualquier tema y producir resultados basados en hechos. 
            No te inventas nada, te esfuerzas al máximo por recopilar datos y hechos para respaldar tu investigación.
            
            Asegúrate de completar el objetivo anterior siguiendo estas normas:
            1/ Debes investigar lo suficiente para recopilar toda la información posible sobre el objetivo.
            2/ Si hay enlaces o artículos relevantes, los analizarás para obtener más información.
            3/ Después de analizar y buscar, plantéate: "¿Hay algo nuevo que deba buscar o analizar basándome en los datos que he recopilado para mejorar la calidad de la investigación?". Si la respuesta es sí, continúa; pero no hagas esto más de 3 veces.
            4/ No debes inventarte nada, solo debes escribir hechos y datos que hayas recopilado.
            5/ En el resultado final, debes incluir todas las referencias y enlaces para respaldar tu investigación. Repito: incluye todas las referencias y enlaces para respaldar tu investigación.
            6/ Siempre consulta primero la web.
            7/ Proporciona toda la información posible, asegurándote de que tu respuesta tenga al menos 500 PALABRAS.
            8/ Sé específico en tu investigación, no te limites a señalar una página web y decir que allí se puede encontrar la información. Para eso estás tú.
            

            Ejemplo de lo que NO debes hacer, estos son solo resúmenes de lo que hay en la web y nada específico, ¡no le dicen nada al usuario!

            1/WIRED - WIRED proporciona las últimas noticias, artículos, fotos, presentaciones y vídeos relacionados con la inteligencia artificial. Fuente: WIRED

            2/Artificial Intelligence News - Este sitio web ofrece las últimas noticias y tendencias de IA, junto con investigaciones de la industria e informes sobre tecnología de IA. Fuente: Artificial Intelligence News
            """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


template = """
      Eres un ghostwriter muy experimentado que destaca en la creación de hilos de Twitter.
Se te proporcionará información y un titular sobre un tema. Tu trabajo es usar esta información y tu propio conocimiento
para escribir un hilo de Twitter atractivo.
El primer tuit del hilo debe tener un gancho y motivar al usuario a seguir leyendo.

Aquí tienes la guía de estilo para escribir el hilo:
1. Voz y tono:
Informativo y claro: Prioriza la claridad y precisión al presentar datos. Frases como "Las investigaciones indican", "Los estudios demuestran" y "Los expertos sugieren" aportan un tono de credibilidad.
Casual y atractivo: Mantén un tono conversacional usando contracciones y un lenguaje cercano. Plantea preguntas ocasionales al lector para mantener su interés.
2. Ambiente:
Educativo: Crea una atmósfera donde el lector sienta que está obteniendo información valiosa o aprendiendo algo nuevo.
Invitador: Usa un lenguaje que anime a los lectores a profundizar más, explorar e iniciar un diálogo.
3. Estructura de las frases:
Longitud variada: Usa una mezcla de puntos concisos para enfatizar y frases explicativas más largas para los detalles.
Frases descriptivas: En lugar de frases imperativas, usa descriptivas para proporcionar información. Por ejemplo, "Elegir un tema puede llevar a..."
4. Estilo de transición:
Secuencial y lógico: Guía al lector a través de la información o los pasos en una secuencia clara y lógica.
Emojis visuales: Los emojis se pueden usar como señales visuales, pero opta por ℹ️ para puntos informativos o ➡️ para indicar continuación.
5. Ritmo y cadencia:
Flujo constante: Asegura un flujo suave de información, transicionando sin problemas de un punto al siguiente.
Datos y fuentes: Introduce ocasionalmente estadísticas, resultados de estudios u opiniones de expertos para respaldar afirmaciones, y ofrece enlaces o referencias para profundizar.
6. Estilos característicos:
Introducciones intrigantes: Comienza los tuits o hilos con un hecho cautivador, una pregunta o una afirmación que capte la atención.
Formato de pregunta y aclaración: Comienza con una pregunta general o una afirmación y sigue con información aclaratoria. Por ejemplo, "¿Por qué es crucial el sueño? Un estudio de la Universidad XYZ señala..."
Uso de '➡️' para continuación: Indica que sigue más información, especialmente útil en hilos.
Resúmenes atractivos: Concluye con un resumen conciso o una invitación a seguir debatiendo para mantener viva la conversación.
Indicadores distintivos para un estilo informativo en Twitter:

Comenzar con hechos y datos: Fundamenta el contenido en información investigada, haciéndolo creíble y valioso.
Elementos atractivos: El uso constante de preguntas y frases claras y descriptivas asegura el compromiso sin depender demasiado de anécdotas personales.
Emojis visuales como indicadores: Los emojis no son solo para conversaciones casuales; se pueden usar eficazmente para marcar transiciones o enfatizar puntos incluso en un contexto informativo.
Conclusiones abiertas: Terminar con preguntas o invitaciones a la discusión puede involucrar a los lectores y fomentar un sentido de comunidad alrededor del contenido.

Últimas instrucciones:
El hilo de Twitter debe tener entre 3 y 10 tuits.
Cada tuit debe comenzar con (número de tuit/longitud total)
No abuses de los hashtags, usa solo uno o dos para todo el hilo.
Usa enlaces con moderación y solo cuando sea realmente necesario, pero cuando lo hagas, ¡asegúrate de incluirlos realmente!
Devuelve solo el hilo, sin otro texto, y haz que cada tuit sea su propio párrafo.
Asegúrate de que cada tuit tenga menos de 220 caracteres.
    Titular del tema: {topic}
    Información: {info}
    """

prompt = PromptTemplate(
    input_variables=["info","topic"], template=template
)

llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    
)



  
twitapi = tweeter()

def tweetertweet(thread):

    tweets = thread.split("\n\n")
   
    #check each tweet is under 280 chars
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm.predict(prompt)[:280]
    #give some spacing between sentances
    tweets = [s.replace('. ', '.\n\n') for s in tweets]

    for tweet in tweets:
        tweet = tweet.replace('**', '')

    response = twitapi.create_tweet(text=tweets[0])
    id = response.data['id']
    tweets.pop(0)
    for i in tweets:
        print("tweeting: " + i)
        reptweet = twitapi.create_tweet(text=i, 
                                    in_reply_to_tweet_id=id, 
                                    )
        id = reptweet.data['id']


  

def main():
    
    # Set page title and icon
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    # Display header 
    st.header("AI research agent :bird:")
    
    # Get user's research goal input
    query = st.text_input("Research goal")

    # Initialize result and thread state if needed
    if not hasattr(st.session_state, 'result'):
        st.session_state.result = None

    if not hasattr(st.session_state, 'thread'):
        st.session_state.thread = None

    # Do research if query entered and no prior result
    if query and (st.session_state.result is None or st.session_state.thread is None):
        st.write("Doing research for ", query)

        # Run agent to generate result
        st.session_state.result = agent({"input": query})
        
        # Generate thread from result
        st.session_state.thread = llm_chain.predict(topic=query, info=st.session_state.result['output'])

    # Display generated thread and result if available
    if st.session_state.result and st.session_state.thread:
        st.markdown(st.session_state.thread)
        
        # Allow tweeting thread
        tweet = st.button("Tweeeeeet")
        
        # Display info on result 
        st.markdown("Twitter thread Generated from the below research")
        st.markdown(st.session_state.result['output'])
    
        if tweet:
            # Tweet thread
            tweetertweet(st.session_state.thread)
            
 

if __name__ == '__main__':
    main()
