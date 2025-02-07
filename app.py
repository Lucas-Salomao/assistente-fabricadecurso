import os
import requests
import json
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Carregar as credenciais e endpoints
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
embedding_model = os.getenv("EMBEDDING_ENGINE")
gpt_engine = os.getenv("GPT_ENGINE")

# Inicializar o cliente Azure OpenAI
azure_openai_client = AzureOpenAI(
    api_key=azure_openai_key,
    api_version="2024-02-15-preview",
    azure_endpoint=azure_openai_endpoint
)

def generate_embeddings(client, text):
    """
    Função para gerar embeddings usando o cliente Azure OpenAI.
    """
    response = client.embeddings.create(
        input=text,
        model=embedding_model
    )
    embeddings = response.model_dump()
    return embeddings['data'][0]['embedding']

def search_documents(vectorised_user_query):
    """
    Função para realizar a busca no Azure Search com os embeddings gerados.
    """
    url = f"{service_endpoint}/indexes/{index_name}/docs/search?api-version=2023-11-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }
    body = {
        "count": True,
        "select": "chunk",
        "vectorQueries": [
            {
                "vector": vectorised_user_query,
                "k": 5,
                "fields": "text_vector",
                "kind": "vector"
            }
        ]
    }
    
    # Enviar a solicitação de busca
    response = requests.post(url, headers=headers, data=json.dumps(body))
    
    if response.status_code == 200:
        return response.json()['value']
    else:
        print("Erro na busca:", response.status_code)
        return []

def get_chat_response(user_query, context):
    """
    Função para obter uma resposta do modelo GPT usando o contexto da pesquisa.
    """
    system_prompt = f"""
    Você é um especialista em educação profissional e deve responder com base nas informações de um estudo de mercado da área de metalmecânica. A resposta deve ser focada em dados e informações específicas sobre o estudo, sem suposições externas.
    Responda com base apenas no estudo fornecido, e se necessário, forneça referências específicas ao conteúdo do estudo.
    # Você deve atuar como um chatbot especializado em fornecer respostas baseadas em um banco de dados de avaliações de estudos de mercado, armazenadas na Solução de Pesquisa de IA do Azure. Suas respostas devem ser exclusivamente baseadas no conteúdo do banco de dados fornecido, sem incluir informações adicionais que não estejam no contexto. Caso não haja uma resposta disponível para a consulta, informe educadamente ao usuário.

    # O contexto será fornecido como uma lista de objetos, cada um representando uma avaliação de estudo de mercado. Cada objeto contém as seguintes informações:
    # "chunk": "Conteúdo da revisão do estudo."
    # "score": "Pontuação de relevância da revisão."
    # A lista contém as três melhores correspondências com base na similaridade de cosseno entre os embeddings da consulta do usuário e as descrições das avaliações.
    
    # Por favor, forneça respostas de forma profissional e natural, sem que o usuário perceba que o sistema está utilizando uma abordagem de recuperação automática de informações. O objetivo é manter uma conversa fluida, como se estivesse interagindo com um especialista humano.
    """

    user_prompt = f"""
    A consulta do usuário é: {user_query}
    O contexto é: {context}
    """
    
    # Obter resposta do GPT com base no contexto
    chat_completions_response = azure_openai_client.chat.completions.create(
        model=gpt_engine,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8
    )
    
    # Retornar a resposta gerada
    return chat_completions_response.choices[0].message.content

# Streamlit interface
def app():
    st.title('Fábrica de Cursos - Metalmecânica')
    
    # Input do usuário
    user_query = st.text_input("Digite sua pergunta sobre o estudo de mercado:", "")
    if st.button("Enviar"):
        if user_query:
            # Gerar embeddings da consulta do usuário
            with st.spinner('Gerando resposta...'):
                vectorised_user_query = generate_embeddings(azure_openai_client, user_query)
                
                # Buscar documentos relacionados
                documents = search_documents(vectorised_user_query)
                
                # Extrair o contexto com base nos documentos encontrados
                context = []
                for doc in documents:
                    context.append({
                        "chunk": doc['chunk'],
                        "score": doc['@search.score']
                    })
                
                # Obter resposta do GPT com base no contexto
                chat_response = get_chat_response(user_query, context)
                
                # Exibir a resposta gerada (apenas a resposta, sem os vetores e documentos)
                st.subheader("Resposta do Assistente da Fábrica:")
                st.write(chat_response)

if __name__ == "__main__":
    app()
