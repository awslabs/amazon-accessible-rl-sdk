============
Experimental
============

.. currentmodule:: a2rl.experimental.lightgpt

``Whatif`` provides an alternate GPT model called ``LightGPT`` that leverages `PyTorch Lightning
<https://pytorch-lightning.readthedocs.io>`_ to simplify model trainings at scale.

.. admonition:: Call for Actions
    :class: important

    We welcome and encourage for your feedbacks to help us refine this experimental API.

Standing on the shoulder of PyTorch Lightning allows ``Whatif`` to be used out-of-the-box in
easy, flexible, scalable and robust manners. At the same time, ``Whatif`` automatically benefits
from new improvements in future PyTorch Lightning release.

.. admonition:: Non-Exhaustive List of New Capabilities

    Here're some of the capabilities that you can immediately access with ``LightGPT``, by virtue of
    the power of transitivity `a2rl -> pytorch-lightning -> capabilities`.

    You can run ``LightGPT`` on a wide range of accelerators which include (as of
    `PyTorch Lightning v1.6.5 <https://pytorch-lightning.readthedocs.io/en/1.6.5/api/pytorch_lightning.accelerators.Accelerator.html>`_)
    CUDA GPU (i.e., the ``P*`` and ``G*`` Amazon EC2 instances), HPU (i.e., the Amazon ``DL1`` EC2
    instances powered by Gaudi accelerators from Intel's Habana Labs), Google's TPU, etc.

    You can also train ``LightGPT`` models on a wide spectrum of system configurations, ranging from
    single node to multiple nodes. Choose a
    :pl:`distributed training algorithms <extensions/strategy.html>` from a wide range of available
    selections: :pl:`Horovod <accelerators/gpu.html#horovod>`,
    :pl:`DeepSpeed <advanced/model_parallel.html#deepspeed>`,
    :pl:`FairScale <extensions/strategy.html#built-in-training-strategies>`,
    :pl:`Bagua <accelerators/gpu.html#bagua>`, etc.

    Instant access to many :pl:`training techniques <advanced/training_tricks.html>` such as
    :pl:`gradients accumulation <advanced/training_tricks.html#accumulate-gradients>`,
    :pl:`gradient clipping <advanced/training_tricks.html#gradient-clipping>`,
    :pl:`stochastic weight averaging <advanced/training_tricks.html#stochastic-weight-averaging>`,
    :pl:`batch-size finder <advanced/training_tricks.html#batch-size-finder>`,
    :pl:`learning-rate finder <advanced/training_tricks.html#learning-rate-finder>`,
    :pl:`16-bit precision <guides/speed.html#mixed-precision-16-bit-training>`, etc.

    ``LightGPT`` has first-class supports to log training metrics to a plethora of
    :pl:`experiment trackers <extensions/logging.html>` such as
    `TensorBoard <https://www.tensorflow.org/tensorboard/>`_ (default behavior),
    `Wandb <https://wandb.ai/>`_, `Comet <https://www.comet.com/>`_, `Neptune
    <https://neptune.ai/>`_, etc.

    There're many more capabilities that ``LightGPT`` automatically inherits from PyTorch Lightning.
    For a comprehensive tour of those capabilities, please visit
    `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io/en/1.6.5/api/pytorch_lightning.accelerators.Accelerator.html>`_.

.. autosummary::
    :toctree: api/
    :nosignatures:

    LightGPTBuilder
