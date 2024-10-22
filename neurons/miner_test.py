
import time
import bittensor as bt

import detection

# import base miner class which takes care of most of the boilerplate
from detection.base.miner import BaseMinerNeuron
from miners.ppl_model import PPLModel

from transformers.utils import logging as hf_logging

from miners.deberta_classifier import DebertaClassifier

hf_logging.set_verbosity(40)


class MinerTest(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    async def forward(
            self, synapse: detection.protocol.TextSynapse
    ) -> detection.protocol.TextSynapse:
        """
        Processes the incoming 'TextSynapse' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (detection.protocol.TextSynapse): The synapse object containing the 'texts' data.

        Returns:
            detection.protocol.TextSynapse: The synapse object with the 'predictions'.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        start_time = time.time()

        input_data = synapse.texts
        bt.logging.info(f"Amount of texts recieved: {len(input_data)}")

        try:
            preds = self.model.predict_batch(input_data)
        except Exception as e:
            bt.logging.error('Couldnt proceed text "{}..."'.format(input_data))
            bt.logging.error(e)
            preds = [0] * len(input_data)

        preds = [[pred] * len(text.split()) for pred, text in zip(preds, input_data)]
        bt.logging.info(f"Made predictions in {int(time.time() - start_time)}s")

        synapse.predictions = preds
        return synapse

    def __init__(self, config=None):
        #super(MinerTest, self).__init__(config=config)

        self.model = DebertaClassifier(foundation_model_path="/models/mistral-7b-v0.1",
                                           model_path="/models/mistral-7b-v0.1",
                                           device="cuda:0")

        self.load_state()

    def predict(self):

        start_time = time.time()
        input_data = "After her work was flagged, Olmsted says she became obsessive about avoiding another accusation. " \
                     "She screen-recorded herself on her laptop doing writing assignments. She worked in Google Docs to " \
                     "track her changes and create a digital paper trail. She even tried to tweak her vocabulary and " \
                     "syntax. “I am very nervous that I would get this far and run into another AI accusation,” says " \
                     "Olmsted, who is on target to graduate in the spring. “I have so much to lose. Nathan Mendoza, " \
                     "a junior studying chemical engineering at the University of California at San Diego, uses " \
                     "GPTZero to prescreen his work. He says the majority of the time it takes him to complete an " \
                     "assignment is now spent tweaking wordings so he isn’t falsely flagged—in ways he thinks make " \
                     "the writing sound worse. Other students have expedited that process by turning to a batch of " \
                     "so-called AI humanizer services that can automatically rewrite submissions to get past AI detectors."


        print(f"Amount of texts recieved: {len(input_data)}")

        try:
            preds = self.model.predict_batch(input_data)
        except Exception as e:
            bt.logging.error('Couldnt proceed text "{}..."'.format(input_data))
            bt.logging.error(e)
            preds = [0] * len(input_data)

        preds = [[pred] * len(text.split()) for pred, text in zip(preds, input_data)]

        print(f"Made predictions: {preds}")

        print(f"Made predictions in {int(time.time() - start_time)}s")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with MinerTest() as miner:
        while True:
            print("MinerTest running...", time.time())

            # Call the `forward` function inside the loop
            miner.predict()
            break
