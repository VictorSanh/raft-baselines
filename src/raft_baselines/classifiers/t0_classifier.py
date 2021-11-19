from json import decoder
from typing import List, Mapping

import datasets
import torch
from transformers import AutoModelForSeq2SeqLM
from sentence_transformers import util
import numpy as np

from raft_baselines.classifiers.in_context_classifier import InContextClassifier
from raft_baselines.utils.tokenizers import TransformersTokenizer
from raft_baselines.utils.embedders import SentenceTransformersEmbedder


class T0Classifier(InContextClassifier):
    def __init__(
        self,
        *args,
        model_type: str = "bigscience/T0_3B",
        **kwargs,
    ) -> None:
        tokenizer = TransformersTokenizer(model_type)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
        if torch.cuda.device_count() > 1:
            self.device = "cuda:0"
            self.model.parallelize()
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
        self.similarity_embedder = SentenceTransformersEmbedder()

        super().__init__(
            *args,
            tokenizer=tokenizer,
            max_tokens=2048,
            **kwargs,
        )

    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        formatted_examples_without_labels = tuple(
            self.format_dict(
                {col: row[col] for col in self.input_cols if col in row},
            )
            for row in self.training_data
        )
        formatted_target = self.format_dict(target)

        # adapted from https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
        target_embedding = self.similarity_embedder((formatted_target,))
        example_embeddings = self.similarity_embedder(formatted_examples_without_labels)

        similarity_scores = util.pytorch_cos_sim(target_embedding, example_embeddings)[
            0
        ]

        sorted_indices = torch.argsort(similarity_scores, descending=True)
        return self.training_data.select(
            sorted_indices[: self.num_prompt_training_examples].tolist()
        )

    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        encoder_inputs = self.tokenizer([prompt for _ in range(len(self.classes))], return_tensors="pt")
        targets = self.tokenizer(
            [
                f" {clas}"
                if not self.add_prefixes
                else f" {self.classes.index(clas) + 1}"
                for clas in self.classes
            ],
            return_tensors="pt",
            padding=True
        )
        model_inputs = {
            **encoder_inputs,
            "labels": targets["input_ids"]
        }

        def split_chunk(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        chunked_model_inputs = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            for input_ids, attention_mask, labels in zip(
                split_chunk(model_inputs["input_ids"], 2),
                split_chunk(model_inputs["attention_mask"],  2),
                split_chunk(model_inputs["labels"],  2),
            )
        ]

        outputs = []
        for chunk in chunked_model_inputs:
            chunk = {
                k: v.to(self.device)
                for k, v in chunk.items()
            }
            with torch.no_grad():
                o = self.model(**chunk).logits
                o = o.cpu()
                outputs.append(o)

        outputs = torch.vstack(outputs)
        masked_logits = targets["attention_mask"].unsqueeze(-1) * outputs
        seq_token_probs = torch.gather(masked_logits, -1, targets["input_ids"].unsqueeze(-1))
        seq_prob = seq_token_probs.squeeze(dim=-1).sum(dim=-1)

        return seq_prob.detach().numpy()
