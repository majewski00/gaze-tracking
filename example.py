import os

import fire
import matplotlib.pyplot as plt

from gaze_tracking import Image, GazeTracker, GazeCollector
from typing import Optional
from os.path import join, exists

import keras
import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate


def euclidean_distance(y_true, y_pred):
    y_true = tf.multiply(y_true, tf.constant([[1920, 1080]], dtype='float32'))
    y_pred = tf.multiply(y_pred, tf.constant([[1920, 1080]], dtype='float32'))
    norm = tf.norm(y_true - y_pred, ord='euclidean', axis=-1)
    return tf.reduce_mean(norm, axis=-1)


def learning_results(hist):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    if list(hist.keys())[3] == 'val_euclidean_distance':
        name = 'Euclidean Distance'
        ax[1].set_title('Eye Tracking Distance Error')
        ax[0].set_title('Eye Tracking Loss Function')
    else:
        name = 'Accuracy'
        ax[1].set_title('Training Accuracy')
        ax[0].set_title('Training Loss Function')

    ax[1].plot(range(len(hist['loss'])), list(hist.values())[1], label=name, color='blue')
    ax[1].plot(range(len(hist['loss'])), list(hist.values())[3], label='Validation ' + name,
               color='red', linestyle=':')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')

    ax[0].plot(range(len(hist['loss'])), hist['loss'], label='Loss Function', color=(0.8, 0.42, 0.97))
    ax[0].plot(range(len(hist['loss'])), hist['val_loss'], label='Validation Loss Function', color=(0.41, 0.01, 0.62))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')

    plt.show()


def main(
        dataset_path: str = None,
        collect_dataset: Optional[bool] = False,
        show_camera: Optional[bool] = False,
        horizontal_data_collection: Optional[bool] = True,
        pretrain_model: Optional[bool] = False,
        resnet_18: Optional[bool] = False,
        load_model: Optional[str] = None,
):
    """
    Example Function with all possible actions in one place (no data analysis).

    Args:
        dataset_path: NOT OPTIONAL - path to dataset directory.
        collect_dataset:
        show_camera:
        horizontal_data_collection:
        pretrain_model:
        resnet_18:
        load_model:

    """
    print("\n\n")
    base_path = os.path.dirname(os.path.realpath(__file__))
    while 1:
        if dataset_path is None:
            inp = input("Enter Dataset Directory Path: ")
            if exists(join(os.path.dirname(os.path.realpath(__file__)), inp)):
                dataset_path = join(base_path, inp)
            else: print('Path do not exists...')
        else:
            break


    if collect_dataset:
        gaze_collector = GazeCollector()
        gaze_collector.collect(dataset_directory=dataset_path, show=show_camera, horizontal=horizontal_data_collection)

    image = Image()
    gaze_tracker = GazeTracker()

    if not exists(join(base_path, "Models")):
        os.makedirs(join(base_path, "Models"))



    if pretrain_model:
        tl_model = gaze_tracker.build_tl_model(resnet_18=resnet_18)
        tl_dataset, tl_size = image.create_dataset(dataset_directory=dataset_path,
                                                   shuffle=True,
                                                   mirror=False,
                                                   YCrCb=False,
                                                   transfer_learn=True)

        print("\nTransfer Learning Dataset size: ", tl_size)
        for i, spec in enumerate(tl_dataset.element_spec):
            print(f"{i}. {(spec.name)} shape: {spec.shape}")
        print('\n')

        BATCH_SIZE = 64
        VALIDATION_SPLIT = 0.15

        train_tl_dataset = tl_dataset.take(int((1 - VALIDATION_SPLIT) * tl_size)).batch(BATCH_SIZE,
                                                                                        drop_remainder=True).repeat()
        val_tl_dataset = tl_dataset.skip(int((1 - VALIDATION_SPLIT) * tl_size)).batch(BATCH_SIZE, drop_remainder=True)

        EPOCHS = 5
        INIT_LR = 2e-4
        MAX_LR = 3e-2
        steps_per_epoch = int((1 - VALIDATION_SPLIT) * tl_size) // BATCH_SIZE
        val_steps = int(VALIDATION_SPLIT * tl_size) // BATCH_SIZE
        clr = CyclicalLearningRate(initial_learning_rate=INIT_LR, maximal_learning_rate=MAX_LR,
                                   scale_fn=lambda x: 1 / (2. ** (x - 1)),
                                   step_size=2 * steps_per_epoch)

        tl_model.compile(optimizer=keras.optimizers.Adam(clr),
                         loss=keras.losses.MeanSquaredError(),
                         metrics=['accuracy'])

        hist = tl_model.fit(train_tl_dataset, epochs=EPOCHS, validation_data=val_tl_dataset,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=val_steps, validation_batch_size=64)
        learning_results(hist.history)

        if input('\nDo you want to save pretrained weights? [y/n] ') == ('y' or 'Y' or 'yes' or 'Yes' or '1'):
            face_resnet18.save_weights(join(base_path, 'Models', f'pretrain_resnet{"18" if resnet_18 else "34"}.h5'))




    ## Model Training

    dataset, size = image.create_dataset(dataset_directory=dataset_path,
                                         shuffle=True,
                                         mirror=True,
                                         YCrCb=False)

    if exists(join(base_path, 'Models', f'pretrain_resnet{"18" if resnet_18 else "34"}.h5')):
        pretrained_weights = join(base_path, 'Models', f'pretrain_resnet{"18" if resnet_18 else "34"}.h5')
    else:
        pretrained_weights = None


    model = gaze_tracker.build_model(resnet_18=resnet_18,
                                     connect_eyes=True,
                                     pretrained_params_path=pretrained_weights
                                     )

    print("Learning Dataset size: ", size)

    if input("\nDo You want to customize training parameters? [y/n] ") == ('y' or 'Y' or 'yes' or 'Yes' or '1'):
        BATCH_SIZE = int(input("    Batch size: "))
        VALIDATION_SPLIT = float(input("    Validation split (e.q. 0.15): "))
        EPOCHS = int(input("    # Epoch: "))
    else:
        BATCH_SIZE = 32
        VALIDATION_SPLIT = 0.15
        EPOCHS = 20

    train_dataset = dataset.take(int((1 - VALIDATION_SPLIT) * size)).batch(BATCH_SIZE,
                                                                                 drop_remainder=True).repeat()
    val_dataset = dataset.skip(int((1 - VALIDATION_SPLIT) * size)).batch(BATCH_SIZE, drop_remainder=True)

    INIT_LR = 1e-4
    MAX_LR = 2e-3
    steps_per_epoch = int((1 - VALIDATION_SPLIT) * size) // BATCH_SIZE
    val_steps = int(VALIDATION_SPLIT * size) // BATCH_SIZE
    clr = CyclicalLearningRate(initial_learning_rate=INIT_LR, maximal_learning_rate=MAX_LR,
                               scale_fn=lambda x: 1 / (2. ** (x - 1)),
                               step_size=2 * steps_per_epoch)


    model.compile(optimizer=keras.optimizers.Adam(clr),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[euclidean_distance])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, start_from_epoch=5,
                                             restore_best_weights=True)

    if load_model is not None:
        model.load_weights(join(base_path, 'Models', load_model))

    hist = model.fit(train_dataset,
                     epochs=EPOCHS,
                     validation_data=val_dataset,
                     callbacks=[callback],
                     batch_size=BATCH_SIZE,
                     steps_per_epoch=steps_per_epoch,
                     validation_steps=val_steps,
                     validation_batch_size=BATCH_SIZE
                     )

    learning_results(hist.history)
    if input('\nDo you want to save pretrained weights? [y/n] ') == ('y' or 'Y' or 'yes' or 'Yes' or '1'):
        model.save_weights(join(base_path, "Models", input("  Model_name (.h5): ")))



if __name__ == "__main__":
    fire.Fire(main)
