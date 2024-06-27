from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings

warnings.filterwarnings("ignore")


class LLaMA:
    def __init__(self, model_path: str = None, gpu: bool = False):
        self.model_path = model_path
        self.model: LLMChain = self.load_model(gpu)

    @staticmethod
    def create_prompt() -> PromptTemplate:
        _default_prompt: str = """
        You are an assistant tasked with taking a natural language \
        query from a user and converting it into a query for a vectorstore. \
        In this process, you strip out information that is not relevant for \
        the retrieval task. Here is the user query: {question}"""

        prompt: PromptTemplate = PromptTemplate(
            input_variables=['question'],
            template=_default_prompt
        )

        return prompt

    def load_model(self, gpu) -> LLMChain:
        """Loads model"""
        callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
        kwargs = {'model_path': MODEL_FILE, 'temperature': 0.5,   'max_tokens': 2000,
                  'top_p': 1, 'callback_manager': callback_manager, 'verbose': True}

        if gpu:
            n_gpu_layers = 40
            n_batch = 512
            kwargs['n_gpu_layers'] = n_gpu_layers,
            kwargs['n_batch'] = n_batch

            model: LlamaCpp = LlamaCpp(**kwargs)

        else:
            model: LlamaCpp = LlamaCpp(**kwargs)

        prompt: PromptTemplate = self.create_prompt()

        llm_chain: LLMChain = LLMChain(
            llm=model,
            prompt=prompt
        )

        return llm_chain

    def inference(self, prompt: str) -> str:
        response: str = self.model.run(prompt)

        return response
