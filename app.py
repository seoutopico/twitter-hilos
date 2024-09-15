import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Query as FastAPIQuery
from twit import tweeter

# Configuración del modelo de lenguaje
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la clave API de OpenAI en las variables de entorno.")

llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=api_key)

# Prompt para generar el contenido de investigación
research_template = """
Eres un investigador de primera clase, capaz de realizar investigaciones detalladas sobre cualquier tema y producir resultados basados en hechos. 
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
       

{query}


"""

research_prompt = PromptTemplate.from_template(research_template)

# Prompt para generar el hilo de Twitter
twitter_template = """
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

Información: {info}

HILO DE TWITTER:
"""

twitter_prompt = PromptTemplate.from_template(twitter_template)

# Creamos las cadenas de investigación y Twitter
research_chain = research_prompt | llm | StrOutputParser()
twitter_chain = twitter_prompt | llm | StrOutputParser()

# Configuración de FastAPI
app = FastAPI()

@app.get("/")
@app.post("/")
async def researchAgent(query: str = FastAPIQuery(None)):
    if query is None:
        return {"error": "Por favor, proporciona una consulta de investigación."}
    # Generar el contenido de la investigación
    research_content = await research_chain.ainvoke({"query": query})
    # Generar el hilo de Twitter
    twitter_thread = await twitter_chain.ainvoke({"info": research_content})
    
    # Publicar el hilo de Twitter
    tweet_result = tweetertweet(twitter_thread)
    
    return {
        "research": research_content, 
        "twitter_thread": twitter_thread,
        "tweet_status": tweet_result
    }


def tweetertweet(thread):
    tweets = thread.split("\n\n")
    
    twitapi = tweeter()  # Get the authenticated Tweepy client
   
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm.predict(prompt)[:280]
    tweets = [s.replace('. ', '.\n\n') for s in tweets]
    for tweet in tweets:
        tweet = tweet.replace('**', '')
    try:
        response = twitapi.create_tweet(text=tweets[0])
        id = response.data['id']
        tweets.pop(0)
        for i in tweets:
            print("tweeting: " + i)
            reptweet = twitapi.create_tweet(text=i, 
                                    in_reply_to_tweet_id=id, 
                                    )
            id = reptweet.data['id']
        return "Tweets posted successfully"
    except Exception as e:
        return f"Error posting tweets: {e}"
