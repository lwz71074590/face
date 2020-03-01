from easydict import EasyDict
from algorithm.fire_detection.NN_model.factory_provider import hparams_factory, model_factory
from algorithm.fire_detection.data_prepare.configure_dataset import config_dataset

def get_model_and_hparams(net_is_using):

    # Get model hparams_conf.
    hparams_model = hparams_factory().get_model_hparams(net_is_using)

    # Get data hparams.
    hparams_data = config_dataset()

    # Merge hparams.
    hparams = merge_hparams(hparams_model, hparams_data)

    # Get model.
    model = model_factory().get_model(net_is_using, hparams)
    return hparams, model


def merge_hparams(hparams1, hparams2):

    hparams = EasyDict()

    for h1 in hparams1:
        hparams[h1] = hparams1[h1]

    for h2 in hparams2:
        hparams[h2] = hparams2[h2]

    return hparams