from __future__ import print_function
import numpy as np
from algorithm.fire_detection.smoke_detection_core.motion_detection import img_to_block


def smoke_classification(sess, model, frames_array, motion_blocks):
    # Use model to classify smoke.
    blocks_num = len(motion_blocks)
    smoke_blocks = []
    if blocks_num > 0:
        block_size = model.hparams.block_size
        all_block_data = np.zeros([blocks_num, model.hparams.sample_sum_frames, block_size, block_size, 3], dtype=np.uint8)
        for index, block in enumerate(motion_blocks):
            x, y = block[0], block[1]
            all_block_data[index, :, :, :, :] = frames_array[:, x:x + block_size, y:y + block_size, :]

        # Standardization. Keep coincident with training model.
        if model.hparams.is_standardization:
            all_block_data = (all_block_data - 128.0) / 128.0

        # Classify.
        argmax_labels = list()
        batch_num = model.hparams.batch_size
        batches = int(blocks_num/batch_num)
        for i in range(batches):
            batch_argmax_labels = sess.run(model.argmax_output,
                                   feed_dict={model.ph_data: all_block_data[i*batch_num:(i+1)*batch_num],
                                              model.ph_is_training: False})
            argmax_labels.append(batch_argmax_labels)
        if blocks_num%batch_num != 0:
            last_batch_data_start_index = batches*batch_num
            last_batch_argmax_labels = sess.run(model.argmax_output,
                                        feed_dict={model.ph_data: all_block_data[last_batch_data_start_index:],
                                                   model.ph_is_training: False})
            argmax_labels.append(last_batch_argmax_labels)
        argmax_labels = np.concatenate(argmax_labels, axis=0)
        smoke_blocks_indexes = np.where(argmax_labels==1)
        smoke_blocks = np.take(motion_blocks, smoke_blocks_indexes, axis=0)
        smoke_blocks = smoke_blocks[0]  # This code is added because smoke_blocks dimension is 3 when I debug.
    return smoke_blocks


def single_frame_detect(sess, model, frame, height, width):

    frames = []
    frames.append(frame)
    frames_array = np.array(frames)

    block_size = model.hparams.block_size

    rows = int(height)
    cols = int(width)
    location_list = img_to_block(rows, cols, model.hparams.block_size)

    motion_blocks = model.motion_detector(frames_array, location_list, block_size)
    smoke_blocks = smoke_classification(sess, model, frames_array, motion_blocks)

    return smoke_blocks





