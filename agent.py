import os
import time
from dotenv import load_dotenv
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

REACT_PROMPT = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
- Action Input must always be a valid JSON object with double quotes only
- Once you have created the issue and received a key (e.g. KAN-5), IMMEDIATELY write Final Answer and STOP
- NEVER call any tool after create_issue succeeds
- NEVER repeat the same action twice
- Final Answer must summarize the created ticket key and title

Begin!

Question: {input}
Thought:{agent_scratchpad}""")


class JiraSupportAgent:
    def __init__(self):
        # 1. Setup Jira
        self.jira = JiraAPIWrapper()
        self.toolkit = JiraToolkit.from_jira_api_wrapper(self.jira)
        self.tools = self.toolkit.get_tools()

        # 2. Setup Groq
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # 3. Setup Agent
        self.agent = create_react_agent(self.llm, self.tools, REACT_PROMPT)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=False
        )

    def create_ticket(self, user_query):
        system_instructions = (
            f"Tu es un ingénieur de support. Ta mission est d'aider l'utilisateur en créant un ticket Jira.\n"
            f"Utilise obligatoirement le projet avec la clé 'KAN'.\n"
            f"Choisis le type de ticket le plus adapté parmi ces types disponibles : Epic, Support, Tache, Incident, Service Request.\n"
            f"- Si c'est une panne ou un problème technique → Incident\n"
            f"- Si c'est une demande d'accès ou de service → Service Request\n"
            f"- Si c'est une question ou aide générale → Support\n"
            f"- Si c'est une grande fonctionnalité → Epic\n"
            f"- Si c'est une tâche simple → Tache\n"
            f"Ne mets jamais de guillemets dans les valeurs des champs Jira.\n"
            f"Requête de l'utilisateur : {user_query}"
        )

        for attempt in range(3):
            try:
                print(f"🚀 Analyse de la requête : {user_query}")
                result = self.executor.invoke({"input": system_instructions})
                return result["output"]
            except Exception as e:
                if "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"⏳ Rate limit, attente {wait}s... (tentative {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    return f"❌ Erreur lors de la création : {str(e)}"

        return "❌ Quota dépassé après 3 tentatives, réessaie dans quelques minutes."


if __name__ == "__main__":
    bot = JiraSupportAgent()

    test_query = "Mon VPN ne se connecte plus depuis ce matin, je suis bloquée pour accéder à la prod."
    result = bot.create_ticket(test_query)
    print(f"\n✅ Résultat : {result}")