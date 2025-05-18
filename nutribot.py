!pip -q install google-genai
!pip show google-generativeai
!pip show google-adk


import os
from google.colab import userdata
OPEN_FOOD_FACTS_URL = os.environ.get("OPEN_FOOD_FACTS_URL", "https://world.openfoodfacts.org/api/v0/product")
print(OPEN_FOOD_FACTS_URL)

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

from google import genai

client = genai.Client()

Model_ID = "gemini-2.0-flash"

!pip install -q google-adk

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import textwrap
from IPython.display import display, Markdown
import requests
import warnings


warnings.filterwarnings("ignore")

from google.generativeai import configure, GenerativeModel


# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response

    # Função auxiliar para exibir texto formatado em Markdown no Colab
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


  ##########################################
# --- Agente 1:Agente buscador de dados nutricionais  --- #
##########################################
def agente_buscador(topico):
  buscador = Agent(
  name = "agente_buscador",
  model = "gemini-2.0-flash",
  description = "Agente que busca dados nutricionais",
  tools = [google_search],
  instruction="""
 Você é um assitente de pesquisa. A sua tarefa é usar a ferramenta de busca do google (google_search).
  para acessar a https://world.openfoodfacts.org/api/v0/product e buscar por dados nutricionais solicitados você deverá
  fazer uma tabela e explicar e um modo simples usando emojis para que o cliente se sinta acolhido 
  """
  )

  entrada_do_agente_buscador = f"Tópico: {topico}"
  dados_nutricionais = call_agent(buscador,entrada_do_agente_buscador)
  return dados_nutricionais

  ################################################
# --- Agente 2: Agente médico --- #
################################################
def agente_planejador(topico,dados_nutricionais_buscados,acao_usuario):
    planejador = Agent(
        name="agente_planejador",
        model="gemini-2.0-flash",
        # Inserir as instruções do Agente Médico #################################################
        instruction="""
        Você é um médicgo com linguagem acessível.Considerando que um paciente digitou na (acao_usuario) sua condição médica
        ,indique uma porção segura e saudável para ele consumir, explicando o porquê dessa quantidade.
        Alternativamente, se a porção usual não for recomendada, gere outras recomendações de alimentos.
        """,
        description="Agente medico",
        tools=[google_search]
    )

    entrada_do_agente_planejador = f"Condição Médica: {acao_usuario}\nAlimento Solicitado: {topico}\nResultados da Busca Inicial: {dados_nutricionais_buscados}"    # Executa o agente
    plano_do_post = call_agent(planejador, entrada_do_agente_planejador)
    return plano_do_post

    ################################################
# --- Agente 3: Agente Chef --- #
################################################
def agente_chefe(topico,acao_usuario):
    planejador = Agent(
        name="agente_chefe",
        model="gemini-2.0-flash",
        # Inserir as instruções do Agente Médico #################################################
        instruction="""
        Você é um chefe com linguagem informal.Considerando que um paciente digitou na (acao_usuario) sua condição médica
        ,indique uma porção segura e saudável para ele consumir, explicando o porquê dessa quantidade.
        Alternativamente, se a porção usual não for recomendada, gere uma nova receita utilizando o alimento desscrito em uma quantidade adequada
        e com ingredientes que beneficiem sua condição.Use o google_search para obter informações sobre os alimentos e as receitas e as doenças.
        """,
        description="Agente medico",
        tools=[google_search]
    )

    entrada_do_agente_chefe = f"Condição Médica: {acao_usuario}\nAlimento Solicitado: {topico}\nResultados da Busca Inicial: {dados_nutricionais_buscados}"    # Executa o agente
    plano_do_chefe = call_agent(planejador, entrada_do_agente_chefe)
    return plano_do_chefe

    ################################################
# --- Agente 4: Agente busca --- #
################################################
def agente_busca_receita(topico,acao_usuario,receitas_geradas):
    planejador = Agent(
        name="agente_busca_receita",
        model="gemini-2.0-flash",
        # Inserir as instruções do Agente Médico #################################################
        instruction="""
        Você é um ajudante que irá buscar no https://www.youtube.com/ receitas relacionadas ao alimento solicitado e à condição médica do usuário.
        Forneça links clicaveis e ou a imagens do video para os vídeos de receitas. Se não encontrar receitas diretamente relacionadas, procure receitas similares
        """,
        description="Agente medico",
        tools=[google_search]
    )

    entrada_do_agente_busca_receita = f"Condição Médica: {acao_usuario}\nAlimento Solicitado: {topico}\nResultados da Busca Inicial: {dados_nutricionais_buscados}"    # Executa o agente
    plano_do_busca_receita = call_agent(planejador, entrada_do_agente_busca_receita)
    return plano_do_busca_receita

    from IPython.display import HTML, display

def exibir_introducao_html():
    html_intro = """
    <div style="background-color: lightblue; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="font-size: 1.9em; color: blue;">🥗 Bem-vindo ao NutriBot ! 🍎</h1>
        <p style="font-size: 1.1em; color: blue;">Seu guia personalizado para uma alimentação saudável.</p>
    </div>
    """
    display(HTML(html_intro))

def exibir_imagem(url_imagem, legenda=None):
    html_imagem = f'<img src="{url_imagem}" alt="{legenda if legenda else "Imagem"}">'
    if legenda:
        html_imagem += f'<p style="font-size: 0.8em; color: #777;">{legenda}</p>'
    display(HTML(html_imagem))

# ... (seu código de importações e definições de agentes) ...

exibir_introducao_html() # Chama a função para exibir a introdução estilizada

# --- Obter o Tópico do Usuário ---
topico = input(" 🍴 Qual alimento você busca informação:\n")
# Inserir lógica do sistema de agentes ################################################
if not topico:
    print("Você esqueceu de digitar o tópico!")
else:
    print(f"Que delicia! Vamos falar sobre {topico}")

    dados_nutricionais_buscados = agente_buscador(topico)
    print("\n 🔍---Resultado do Agente 1 (Buscador de nutrientes)---")
    exibir_imagem("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjW0HqSNMQCLhXfw4eMt4nIRTsH-nYHQr_GsK3SGqSm2G6_UQVUhJGsDhFfcBr6F2BlSvLJZrbkisBWG1im24R9oV09fEDDCTsILd4d4vcw4Hhexzh2Sr2TAZFvQH-nOkDu8paUXwkujlc/s1600/alimentos.gif", "Uma maçã animada") # Chamada correta da função

    display(to_markdown(dados_nutricionais_buscados))
    print("---------------------------------------------")


    print("\n 🩺---Resultado do Agente 2 (Agente Médico)---\n")
    acao_usuario = input("🩺 Você possui alguma condição médica:\n Se a resposta for sim digite qual:\n")
    if acao_usuario == "não":
      exibir_imagem("https://gifdb.com/images/high/medica-mr-bean-rowan-atkinson-thumbs-up-puneakb9gv2y4s21.gif", "médico fazendo ok")
    else:
        exibir_imagem("data:image/jpeg;base64,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", "médico anotando")

    plano_de_post = agente_planejador(topico,dados_nutricionais_buscados,acao_usuario)
    display(to_markdown(plano_de_post))
    print("---------------------------------------------")

    receitas_geradas = agente_chefe(topico,acao_usuario)
    print("\n 👨‍🍳---Resultado do Agente 3 (Agente Chef)---\n")
    exibir_imagem("https://www.socialistamorena.com.br/wp-content/uploads/2016/02/cozinhar.gif")
    display(to_markdown(receitas_geradas))
    print("---------------------------------------------")

    videos_de_receitas_geradas = agente_busca_receita(topico,acao_usuario,receitas_geradas)
    print("\n 🍳---Resultado do Agente 4 (Agente Buscador de Receitas)---\n")
    display(to_markdown(videos_de_receitas_geradas))
    print("---------------------------------------------")
