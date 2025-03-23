import logging
from .jhmdb_eval import do_jhmdb_evaluation


def jhmdb_evaluation(dataset, predictions, output_folder, **kwargs):
    logger = logging.getLogger("alphaction.inference")
    logger.info("performing jhmdb evaluation.")
    return do_jhmdb_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        metric=kwargs.get('metric', 'frame_ap'),
        save_csv=kwargs.get('save_csv', False)
    )