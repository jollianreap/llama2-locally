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
        You are an assistant tasked with getting an answer in natural language. Your task is to answer as shortly as you can.  Here is the user query: {question}"""

        prompt: PromptTemplate = PromptTemplate(
            input_variables=['question'],
            template=_default_prompt
        )

        return prompt

    def load_model(self, gpu) -> LLMChain:
        """Loads model"""
        callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
        kwargs = {'model_path': self.model_path, 'temperature': 0.5,   'max_tokens': 2000,
                  'top_p': 1, 'callback_manager': callback_manager, 'verbose': True}

        if gpu is True:
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
