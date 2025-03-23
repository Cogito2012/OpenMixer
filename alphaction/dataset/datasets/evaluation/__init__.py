from alphaction.dataset import datasets

from .ava import ava_evaluation
from .jhmdb import jhmdb_evaluation
from .ucf24 import ucf24_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.Ava):
        return ava_evaluation(**args)
    elif isinstance(dataset, datasets.Jhmdb):
        return jhmdb_evaluation(**args)
    elif isinstance(dataset, datasets.UCF24):
        return ucf24_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
