import logging
from .ava_eval import do_ava_evaluation


def ava_evaluation(dataset, predictions, output_folder, **kwargs):
    logger = logging.getLogger("alphaction.inference")
    logger.info("performing ava evaluation.")
    return do_ava_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        metric=kwargs.get('metric', 'frame_ap'),
        save_csv=kwargs.get('save_csv', False)
    )
