import runpy
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from .greedy_search import OVLLMGreedy
import numpy as np

class Continuation:
    def __init__(self, continuation):
        assert(type(continuation) is list)
        batch_size = len(continuation)
        self.text = continuation
        self.tokens = []
        self.logprobs = np.full((batch_size,), 0, dtype=np.float32)
        self.is_greedy = np.full((batch_size,), True)        

@register_model("ovllm")
class OVLLMModel(LM):
    def __init__(self, path, nbatch=1, nbeam = 0, prec = 'bf16', **kwargs):
        super().__init__()
        self.ovllm = OVLLMGreedy(path, prec)
        self.nbatch = nbatch

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """

        if not requests:
            return []

        res = []
        total_num = len(requests)
        pbar = tqdm(total=total_num, disable=disable_tqdm)
        for i0 in range(0, total_num, self.nbatch):
            i1 = min(i0 + self.nbatch, total_num)
            conexts = [req.args[0] for req in requests[i0:i1]]
            continuations = [req.args[1] for req in requests[i0:i1]]

            c = Continuation(continuations)

            self.ovllm.generate(conexts, 10, beam_size = 0, continuation = c)

            for logprob, is_greedy in zip(c.logprobs, c.is_greedy):
                res.append((logprob, is_greedy))
            
            pbar.n = i1
            pbar.refresh()

        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "generate_until not yet supported for ovllm models"
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for ovllm models"
        )

if __name__ == "__main__":
    runpy.run_module('lm_eval', run_name='__main__')
