import logging
import os

from dataclasses import dataclass, field
from typing import Optional

from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionConfig, SentencePredictionTask
from fairseq.data import Dictionary
from omegaconf import II

logger = logging.getLogger(__name__)

@dataclass
class GlueConfig(SentencePredictionConfig):
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")
    mask_prob: float = field(default=0.15)

@register_task("glue", dataclass=GlueConfig)
class GlueTask(SentencePredictionTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @classmethod
    def load_dictionary(cls, filename, extra_mask_tokens=False, required_batch_size_multiple=1):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        
        if extra_mask_tokens:
            dictionary.add_symbol("<mask>")
            for i in range(100):
                dictionary.add_symbol(f"<mask_{i}>")

        dictionary.pad_to_multiple_(required_batch_size_multiple)

        return dictionary
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input0", "dict.txt"),
            extra_mask_tokens=True,
            required_batch_size_multiple=cfg.required_batch_size_multiple,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(
                os.path.join(cfg.data, "label", "dict.txt"),
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(cfg, data_dict, label_dict)