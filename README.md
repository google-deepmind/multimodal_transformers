# Multimodal Transformers

This code runs inference with the multimodal transformer models described in "Decoupling the Role of Data,
Attention, and Losses in Multimodal Transformers". Our models can be used to
score if an image-text pair match. Please see our paper for more details.
This code release consists of a colab to extract image and language features
and input them into our transformer models.  Transformer models are stored on
tfhub.


Please see the tables below for details of models which we have released via tfhub.

Name                                 | Training Dataset                    | ITM            | MRM | MLM | Heads | Layers | Att. Type             | FineTuned | Notes
------------------------------------ | ----------------------------------- | -------------- | --- | --- | ----- | ------ | --------------------- | --------- | -----
data_cc (base)                       | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_sbu                             | SBU                                 | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_vg                              | Visual Genome                       | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_mscoco                          | MSCOCO                              | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_mscoco-narratives               | MSCOCO Narratives                   | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_oi-narratives                   | OI Narratives                       | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_combined-instance               | All (instance sampling)             | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_combined-dataset                | All (dataset sampling)              | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_uniter-instance                 | Uniter datasets (instance sampling) | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_uniter-dataset                  | Uniter datasets (dataset sampling)  | Classification | Y   | Y   | 12    | 6      | Merged                | N         |
data_cc-with-bert                    | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Merged                | N         | Language initialised with BERT
loss_itm_mrm                         | Conceptual Captions                 | Classification | Y   | N   | 12    | 6      | Merged                | N         |
loss_itm_mlm                         | Conceptual Captions                 | Classification | N   | Y   | 12    | 6      | Merged                | N         |
loss_single-modality-contrastive32   | Conceptual Captions                 | Contrastive    | Y   | Y   | 12    | 6      | Sing. Modality        | N         |
loss_single-modality-contrastive1024 | Conceptual Captions                 | Contrastive    | Y   | Y   | 12    | 6      | Sing. Modality        | N         |
loss_v1-contrastive32                | Conceptual Captions                 | Contrastive    | Y   | Y   | 12    | 1      | Merged                | N         |
architecture_heads1-768              | Conceptual Captions                 | Classification | Y   | Y   | 1     | 6      | Merged                | N         |
architecture_heads3-256              | Conceptual Captions                 | Classification | Y   | Y   | 3     | 6      | Merged                | N         |
architecture_heads6-64               | Conceptual Captions                 | Classification | Y   | Y   | 6     | 6      | Merged                | N         |
architecture_heads18-64              | Conceptual Captions                 | Classification | Y   | Y   | 18    | 6      | Merged                | N         |
architecture_vilbert-1block          | Conceptual Captions                 | Classification | Y   | Y   | 12    | 1      | Merged                | N         |
architecture_vilbert-2block          | Conceptual Captions                 | Classification | Y   | Y   | 12    | 2      | Merged                | N         |
architecture_vilbert-4block          | Conceptual Captions                 | Classification | Y   | Y   | 12    | 4      | Merged                | N         |
architecture_vilbert-12block         | Conceptual Captions                 | Classification | Y   | Y   | 12    | 12     | Merged                | N         |
architecture_single-modality         | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Sing. Modality        | N         |
architecture_mixed-modality          | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Mix Modality          | N         | 5 single modality layers and 1 merged layer
architecture_single-stream           | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Single Stream         | N         |
architecture_language-q-12           | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Asymmetric (language) | N         |
architecture_image-q-12              | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Asymmetric (image)    | N         |
architecture_language-q-24           | Conceptual Captions                 | Classification | Y   | Y   | 24    | 6      | Asymmetric (language) | N         |
architecture_image-q-24              | Conceptual Captions                 | Classification | Y   | Y   | 24    | 6      | Asymmetric (image)    | N         |
architecture_single-modality-hloss   | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Single modality       | N         | Includes ITM loss after every layer
data-ft_sbu                          | SBU                                 | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_vg                           | Visual Genome                       | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_mscoco                       | MSCOCO                              | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_mscoco-narratives            | MSCOCO Narratives                   | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_oi-narratives                | OI Narratives                       | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_cc                           | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_combined-instance            | All (instance sampling)             | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_combined-dataset             | All (dataset sampling)              | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_uniter-instance              | Uniter datasets (instance sampling) | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
data-ft_uniter-dataset               | Uniter datasets (dataset sampling)  | Classification | Y   | Y   | 12    | 6      | Merged                | Y         |
architecture-ft_single-modality      | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Sing. Modality        | Y         |
architecture-ft_single-stream        | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Single Stream         | Y         |
architecture-ft_language-q-12        | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Asymmetric (language) | Y         |
architecture-ft_image-q-12           | Conceptual Captions                 | Classification | Y   | Y   | 12    | 6      | Asymmetric (image)    | Y         |
architecture-ft_language-q-24        | Conceptual Captions                 | Classification | Y   | Y   | 24    | 6      | Asymmetric (language) | Y         |
architecture-ft_image-q-24           | Conceptual Captions                 | Classification | Y   | Y   | 24    | 6      | Asymmetric (image)    | Y         |

In addition to our transformer models, we also release our baseline models.  See details of our baseline models in the chart below:

| Name                                  | ITM            | Bert Initialisation | FineTuned |
|---------------------------------------|----------------|---------------------|-----------|
| baseline_baseline                     | Contrastive    | Yes                 | N         |
| baseline_baseline-cls                 | Classification | No                  | N         |
| baseline_baseline-no-bert-transfer    | Contrastive    | No                  | N         |
| baseline-ft_baseline                  | Contrastive    | Yes                 | Y         |
| baseline-ft_baseline-cls              | Classification | No                  | Y         |
| baseline-ft_baseline-no-bert-transfer | Contrastive    | No                  |

## Installation

You do not need to install anything!  You should be able to run all code
from our released colab.

## Usage

You can run an image and text pair through our module and see if the image and
text pair match.

```python
import tensorflow.compat.v1 as tf import tensorflow_hub as hub
m =
hub.Module('https://tfhub.dev/deepmind/mmt/architecture-ft_image-q-12/1')
```

Inference:

```python
output = model.signatures['default'](**inputs)
score = tf.nn.softmax(output['output']).numpy()[0]
```

where `score` indicates if an image-text pair match (`1` indicates a perfect
match).  Inputs is a dictionary with the following keys:

* `image/bboxes`: Coordinates of detected image bounding boxes.

* `image/detection_features`: Features from image detector.

* `image/padding_mask`: Indicator if image features are padded.

* `masked_tokens`: Text tokens

* `text/segment_ids`: Indicates sentence segment.  (Since we train with one sentencce this will always be 0.)

* `text/token_ids`:  Indicates which words tokens belong to.  (We use a tokenizer which can break one word into multiple tokens).

* `text/padding_mask`: Indicator if text features are padded.

Please see our colab linked for details on pre-processing.
You will need to use the detector released in our colab for good results.

## Citing this work

If you use this model in your research please cite:

[1] Lisa Anne Hendricks, John Mellor, Rosalia Schneider, Jean-Baptiste Alayrac,
and Aida Nematzadeh.
[Decoupling the Role of Data, Attention, and Losses
in Multimodal Transformers](https://arxiv.org/pdf/2102.00529.pdf),
TACL 2021.

## Disclaimer

This is not an official Google product.
