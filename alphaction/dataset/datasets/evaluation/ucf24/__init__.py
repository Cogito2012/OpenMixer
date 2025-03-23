import logging
from .ucf24_eval import do_ucf24_evaluation


def ucf24_evaluation(dataset, predictions, output_folder, **kwargs):
    logger = logging.getLogger("alphaction.inference")
    logger.info("performing UCF24 evaluation.")
    return do_ucf24_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        metric=kwargs.get('metric', 'frame_ap'),
        save_csv=kwargs.get('save_csv', False)
    )