import scipy.misc
import numpy as np
import time
import os
import cv2


def unstack(np_array):
    new_list = []
    for i in range(np_array.shape[0]):
        temp_list = np_array[i]
        new_list.append(temp_list)
    return new_list


def sample_generator(num_generations, sess, same_images, dropout_rate, dropout_rate_value, data, batch_size, file_name,
                     conditional_inputs, input_a,
                     support_input, input_b,
                     y_input_i, input_y_i,
                     y_input_j, input_y_j,
                     y_global_input_i, input_global_y_i,
                     y_global_input_j, input_global_y_j,
                     classes_number, classes,
                     selected_classes, classes_selected,
                     support_number, number_support,
                     z_input, z_input_2,
                     z_vectors, z_vectors_2,
                     training_phase, is_training,
                     z1z2_training, training_z1z2):
    input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
        same_images, feed_dict={
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: 1,
            z_input: batch_size * [z_vectors[0]],
            z_input_2: batch_size * [z_vectors_2[0]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]

    support_num = support_images.shape[-4]
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    recons_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                         input_images.shape[-1]))

    refer_trans_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                       input_images.shape[-1]))

    support_list = np.zeros(shape=(
    batch_size, num_generations, support_images.shape[-4], support_images.shape[-3], support_images.shape[-2],
    support_images.shape[-1]))

    height = generated.shape[-3]

    is_interpolation = False
    num_interpolation = num_generations

    if is_interpolation:
        for i in range(num_interpolation):
            # z_vectors[i] = z_vectors[0]*(1-i*1e-1) + z_vectors[-1]*(i*1e-1)
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={
                    input_a: conditional_inputs,
                    input_b: support_input,
                    input_y_i: y_input_i,
                    input_y_j: y_input_j,
                    input_global_y_i: y_global_input_i,
                    input_global_y_j: y_global_input_j,
                    classes: classes_number,
                    classes_selected: selected_classes,
                    number_support: support_number,
                    z_input: batch_size * [z_vectors[i]],
                    z_input_2: batch_size * [z_vectors_2[i]],
                    dropout_rate: dropout_rate_value,
                    training_phase: is_training,
                    z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            recons_images_list[:, i] = recons_image
            generated_list[:, i] = generated
            support_list[:, i] = support_images

    else:
        for i in range(num_generations):
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={z_input: batch_size * [z_vectors[i]],
                                        input_a: conditional_inputs,
                                        input_b: support_input,
                                        input_y_i: y_input_i,
                                        input_y_j: y_input_j,
                                        input_global_y_i: y_global_input_i,
                                        input_global_y_j: y_global_input_j,
                                        classes: classes_number,
                                        classes_selected: selected_classes,
                                        number_support: support_number,
                                        z_input: batch_size * [z_vectors[i]],
                                        z_input_2: batch_size * [z_vectors_2[i]],
                                        dropout_rate: dropout_rate_value,
                                        training_phase: is_training,
                                        z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            recons_images_list[:, i] = recons_image
            refer_trans_list[:, i] = refer_trans

    input_images, support_images, generated, reconstructed, refer_translation = data.reconstruct_original(
        input_images_list), data.reconstruct_original(
        support_list), data.reconstruct_original(generated_list), data.reconstruct_original(
        recons_images_list), data.reconstruct_original(refer_trans_list)
    images = np.zeros(
        shape=(
        batch_size, 1 + support_num + 1 + 1 + num_generations, support_images.shape[-3], support_images.shape[-2],
        support_images.shape[-1]))

    # for i in range(num_generations):
    #     print('index of image',i)
    #     print('image difference',np.sum(generated[0]-generated[i]))

    x_total_images = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + num_generations), image_size, image_size, channel])
    y_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), classes_number])
    y_few_shot_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), selected_classes])

    for k in range(1 + support_num + 1 + 1 + num_generations):
        if k == 0:
            images[:, k] = input_images[:, 0]
            x_total_images[:batch_size] = input_images[:, 0]
            y_total_image[:batch_size] = y_label
            y_few_shot_total_image[:batch_size] = few_shot_y_label
        elif k > 0 and k < support_num + 1:
            images[:, k] = support_images[:, 0, k - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = support_images[:, 0, k - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 1:
            images[:, k] = reconstructed[:, 0]
            x_total_images[batch_size * k:batch_size * (k + 1)] = reconstructed[:, 0]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 2:
            images[:, k] = refer_translation[:, 0]
        else:
            images[:, k] = generated[:, k - 1 - support_num - 1 - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = generated[:, k - 1 - support_num - 1 - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label

    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    if is_interpolation:
        csv_file_name = file_name.split('.')[0]
        # similarities_list = np.reshape(similarities_list,[batch_size,support_num*num_generations])
        # for i in range(batch_size):
        #     np.savetxt(csv_file_name+'f_{}.txt'.format(i), f_encode_z_list[i], delimiter='\t', newline='\r\n')
        #     np.savetxt(csv_file_name+'v_{}.txt'.format(i), similarities_list[i], delimiter='\t', newline='\r\n')
    # scipy.misc.imsave(file_name, images)

    scipy.misc.imsave(file_name, images)

    # csv_file_name = file_name.split('.')[0]
    # for i in range(batch_size):
    #     np.savetxt(csv_file_name+'sim_{}.txt'.format(i), similarities_list[i], delimiter='\t', newline='\r\n')

    return x_total_images, y_total_image, y_few_shot_total_image


def find_classes(root_dir):
    retour = []
    print('1', root_dir)
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                # if (f.endswith("jpg")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return len(retour)


def sample_generator_store_into_file(num_generations, sess, same_images, dropout_rate, dropout_rate_value, data,
                                     batch_size, file_name,
                                     conditional_inputs, input_a,
                                     support_input, input_b,
                                     y_input_i, input_y_i,
                                     y_input_j, input_y_j,
                                     y_global_input_i, input_global_y_i,
                                     y_global_input_j, input_global_y_j,
                                     classes_number, classes,
                                     selected_classes, classes_selected,
                                     support_number, number_support,
                                     z_input, z_input_2,
                                     z_vectors, z_vectors_2,
                                     training_phase, is_training,
                                     z1z2_training, training_z1z2):
    input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
        same_images, feed_dict={
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: 1,
            z_input: batch_size * [z_vectors[0]],
            z_input_2: batch_size * [z_vectors_2[0]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]

    support_num = support_images.shape[-4]
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    support_list = np.zeros(shape=(
        batch_size, num_generations, support_images.shape[-4], support_images.shape[-3], support_images.shape[-2],
        support_images.shape[-1]))

    recons_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                         input_images.shape[-1]))

    refer_trans_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                       input_images.shape[-1]))

    height = generated.shape[-3]

    is_interpolation = False
    num_interpolation = num_generations

    if is_interpolation:
        for i in range(num_interpolation):
            # z_vectors[i] = z_vectors[0]*(1-i*1e-1) + z_vectors[-1]*(i*1e-1)
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={
                    input_a: conditional_inputs,
                    input_b: support_input,
                    input_y_i: y_input_i,
                    input_y_j: y_input_j,
                    input_global_y_i: y_global_input_i,
                    input_global_y_j: y_global_input_j,
                    classes: classes_number,
                    classes_selected: selected_classes,
                    number_support: support_number,
                    z_input: batch_size * [z_vectors[i]],
                    z_input_2: batch_size * [z_vectors_2[i]],
                    dropout_rate: dropout_rate_value,
                    training_phase: is_training,
                    z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            recons_images_list[:, i] = recons_image
            support_list[:, i] = support_images
            # similarities_list[:, i] = similarities
            # f_encode_z_list[:, i] = f_encode_z
            # matching_feature_list[:,i] = matching_feature

    else:
        for i in range(num_generations):
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={z_input: batch_size * [z_vectors[i]],
                                        input_a: conditional_inputs,
                                        input_b: support_input,
                                        input_y_i: y_input_i,
                                        input_y_j: y_input_j,
                                        input_global_y_i: y_global_input_i,
                                        input_global_y_j: y_global_input_j,
                                        classes: classes_number,
                                        classes_selected: selected_classes,
                                        number_support: support_number,
                                        z_input: batch_size * [z_vectors[i]],
                                        z_input_2: batch_size * [z_vectors_2[i]],
                                        dropout_rate: dropout_rate_value,
                                        training_phase: is_training,
                                        z1z2_training: training_z1z2})

            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            recons_images_list[:, i] = recons_image
            refer_trans_list[:, i] = refer_trans

    input_images, support_images, generated, reconstructed, refer_translation = data.reconstruct_original(
        input_images_list), data.reconstruct_original(
        support_list), data.reconstruct_original(generated_list), data.reconstruct_original(
        recons_images_list), data.reconstruct_original(refer_trans_list)
    images = np.zeros(
        shape=(
            batch_size, 1 + support_num + 1 + 1 + num_generations, support_images.shape[-3], support_images.shape[-2],
            support_images.shape[-1]))

    x_total_images = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + num_generations), image_size, image_size, channel])
    y_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), classes_number])
    y_few_shot_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), selected_classes])

    for k in range(1 + support_num + 1 + 1 + num_generations):
        if k == 0:
            images[:, k] = input_images[:, 0]
            x_total_images[:batch_size] = input_images[:, 0]
            y_total_image[:batch_size] = y_label
            y_few_shot_total_image[:batch_size] = few_shot_y_label
        elif k > 0 and k < support_num + 1:
            images[:, k] = support_images[:, 0, k - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = support_images[:, 0, k - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 1:
            images[:, k] = reconstructed[:, 0]
            x_total_images[batch_size * k:batch_size * (k + 1)] = reconstructed[:, 0]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 2:
            images[:, k] = refer_translation[:, 0]
        else:
            images[:, k] = generated[:, k - 1 - support_num - 1 - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = generated[:, k - 1 - support_num - 1 - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label

    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    testing_num = 30

    image_size = generated.shape[-3]
    for j in range(batch_size):
        current_path_quality = file_name.split('//')[0] + '_forquality' + '/{}/'.format(
            np.argmax(y_global_input_i[:, j]))
        current_path_classifier = file_name.split('//')[0] + '_forclassifier' + '/{}/'.format(
            np.argmax(y_global_input_i[:, j]))
        if not os.path.exists(current_path_quality):
            os.makedirs(current_path_quality)

        if not os.path.exists(current_path_classifier):
            os.makedirs(current_path_classifier)

        if not os.path.exists(current_path_quality) and not os.path.exists(current_path_classifier):
            continue

        for k in range(1 + support_number + num_generations + 1 + 1):
            # if find_classes(current_path_classifier) == (num_generations + support_number + testing_num) and find_classes(current_path_quality) == num_generations:
            #     break
            # if (find_classes(current_path_classifier) = (num_generations + support_number + testing_num))
            #### resize ####
            current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
            if (current_iamge.shape[-2] < 128):
                current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)

            ##### generating images for evaluating the few-shot classifier with generated images
            ###########

            ###########

            if 0 < k < support_num:
                #### classifier ####
                if (find_classes(current_path_classifier) < (num_generations + support_number + testing_num)):
                    current_name_classifier = current_path_classifier + file_name.split('//')[-1].split('.png')[
                        0] + '_sample{}.png'.format(k)
                    scipy.misc.imsave(current_name_classifier, current_iamge)

            elif k > support_num + 1 + 1:
                #### classifier ####
                if (find_classes(current_path_classifier) < (num_generations + support_number + testing_num)):
                    current_name_classifier = current_path_classifier + file_name.split('//')[-1].split('.png')[
                        0] + '_sample{}.png'.format(k)
                    scipy.misc.imsave(current_name_classifier, current_iamge)

                #### quality ####
                if (find_classes(current_path_quality) < num_generations):
                    current_name_quality = current_path_quality + file_name.split('//')[-1].split('.png')[
                        0] + '_sample{}.png'.format(k)
                    scipy.misc.imsave(current_name_quality, current_iamge)
    return x_total_images, y_total_image, y_few_shot_total_image


def sample_generator_store_into_file_fid_generation_real(num_generations, sess, same_images, dropout_rate,
                                                         dropout_rate_value, data,
                                                         batch_size, file_name,
                                                         conditional_inputs, input_a,
                                                         support_input, input_b,
                                                         y_input_i, input_y_i,
                                                         y_input_j, input_y_j,
                                                         y_global_input_i, input_global_y_i,
                                                         y_global_input_j, input_global_y_j,
                                                         classes_number, classes,
                                                         selected_classes, classes_selected,
                                                         support_number, number_support,
                                                         z_input, z_input_2,
                                                         z_vectors, z_vectors_2,
                                                         training_phase, is_training,
                                                         z1z2_training, training_z1z2):
    ##### the number of using samples
    samples_each_category = data.general_classification_samples
    each_samples_generation = int(num_generations / samples_each_category)
    total_class = data.testing_classes

    # ##### the number of using samples
    # z_vectors = np.random.uniform(-1., 1., size=(num_generations, 128))
    # index = np.random.choice(num_generations, size=2, replace=False)
    # fixed_1 =  z_vectors[index[0]]
    # fixed_2 =  z_vectors[index[1]]
    # steps = 1 / num_generations
    # for i in range(num_generations):
    #     if i ==0:
    #         z_vectors[i] = fixed_2
    #     elif i==num_generations-1:
    #         z_vectors[i] = fixed_1
    #     else:
    #         current_steps = i * steps
    #         z_vectors[i] = current_steps*fixed_1 + (1 - current_steps)*fixed_2

    #### interpolation
    input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
        same_images, feed_dict={
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: 1,
            z_input: batch_size * [z_vectors[0]],
            z_input_2: batch_size * [z_vectors_2[0]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})
    # input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
    #             same_images, feed_dict={z_input: z_vectors[:, 0],
    #                                     input_a: conditional_inputs,
    #                                     input_b: support_input,
    #                                     input_y_i: y_input_i,
    #                                     input_y_j: y_input_j,
    #                                     input_global_y_i: y_global_input_i,
    #                                     input_global_y_j: y_global_input_j,
    #                                     classes: classes_number,
    #                                     classes_selected: selected_classes,
    #                                     number_support: support_number,
    #                                     z_input: z_vectors[:, 0],
    #                                     z_input_2: z_vectors_2[:,0],
    #                                     dropout_rate: dropout_rate_value,
    #                                     training_phase: is_training,
    #                                     z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]

    support_num = support_images.shape[-4]
    input_images_list = np.zeros(
        shape=(batch_size, each_samples_generation, input_images.shape[-3], input_images.shape[-2],
               input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, each_samples_generation, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    support_list = np.zeros(shape=(
        batch_size, each_samples_generation, support_images.shape[-4], support_images.shape[-3],
        support_images.shape[-2],
        support_images.shape[-1]))

    recons_images_list = np.zeros(
        shape=(batch_size, each_samples_generation, input_images.shape[-3], input_images.shape[-2],
               input_images.shape[-1]))

    refer_trans_list = np.zeros(
        shape=(batch_size, each_samples_generation, input_images.shape[-3], input_images.shape[-2],
               input_images.shape[-1]))

    height = generated.shape[-3]

    is_interpolation = False
    num_interpolation = num_generations

    if is_interpolation:
        for i in range(num_interpolation):
            # z_vectors[i] = z_vectors[0]*(1-i*1e-1) + z_vectors[-1]*(i*1e-1)
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={
                    input_a: conditional_inputs,
                    input_b: support_input,
                    input_y_i: y_input_i,
                    input_y_j: y_input_j,
                    input_global_y_i: y_global_input_i,
                    input_global_y_j: y_global_input_j,
                    classes: classes_number,
                    classes_selected: selected_classes,
                    number_support: support_number,
                    z_input: batch_size * [z_vectors[i]],
                    z_input_2: batch_size * [z_vectors_2[i]],
                    dropout_rate: dropout_rate_value,
                    training_phase: is_training,
                    z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            recons_images_list[:, i] = recons_image
            support_list[:, i] = support_images
            # similarities_list[:, i] = similarities
            # f_encode_z_list[:, i] = f_encode_z
            # matching_feature_list[:,i] = matching_feature

    else:
        for i in range(each_samples_generation):
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={z_input: batch_size * [z_vectors[i]],
                                        input_a: conditional_inputs,
                                        input_b: support_input,
                                        input_y_i: y_input_i,
                                        input_y_j: y_input_j,
                                        input_global_y_i: y_global_input_i,
                                        input_global_y_j: y_global_input_j,
                                        classes: classes_number,
                                        classes_selected: selected_classes,
                                        number_support: support_number,
                                        z_input: batch_size * [z_vectors[i]],
                                        z_input_2: batch_size * [z_vectors_2[i]],
                                        dropout_rate: dropout_rate_value,
                                        training_phase: is_training,
                                        z1z2_training: training_z1z2})
            # input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
            #     same_images, feed_dict={z_input: z_vectors[:, i],
            #                             input_a: conditional_inputs,
            #                             input_b: support_input,
            #                             input_y_i: y_input_i,
            #                             input_y_j: y_input_j,
            #                             input_global_y_i: y_global_input_i,
            #                             input_global_y_j: y_global_input_j,
            #                             classes: classes_number,
            #                             classes_selected: selected_classes,
            #                             number_support: support_number,
            #                             z_input: z_vectors[:, i],
            #                             z_input_2: z_vectors_2[:,i],
            #                             dropout_rate: dropout_rate_value,
            #                             training_phase: is_training,
            #                             z1z2_training: training_z1z2})

            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            recons_images_list[:, i] = recons_image
            refer_trans_list[:, i] = refer_trans

    input_images, support_images, generated, reconstructed, refer_translation = data.reconstruct_original(
        input_images_list), data.reconstruct_original(
        support_list), data.reconstruct_original(generated_list), data.reconstruct_original(
        recons_images_list), data.reconstruct_original(refer_trans_list)

    images = np.zeros(
        shape=(
            batch_size, 1 + support_num + 1 + 1 + each_samples_generation, support_images.shape[-3],
            support_images.shape[-2],
            support_images.shape[-1]))

    x_total_images = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + each_samples_generation), image_size, image_size, channel])
    y_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + each_samples_generation), classes_number])
    y_few_shot_total_image = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + each_samples_generation), selected_classes])

    for k in range(1 + support_num + 1 + 1 + each_samples_generation):
        if k == 0:
            images[:, k] = input_images[:, 0]
            x_total_images[:batch_size] = input_images[:, 0]
            y_total_image[:batch_size] = y_label
            y_few_shot_total_image[:batch_size] = few_shot_y_label
        elif k > 0 and k < support_num + 1:
            images[:, k] = support_images[:, 0, k - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = support_images[:, 0, k - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 1:
            images[:, k] = reconstructed[:, 0]
            x_total_images[batch_size * k:batch_size * (k + 1)] = reconstructed[:, 0]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 2:
            images[:, k] = refer_translation[:, 0]
        else:
            images[:, k] = generated[:, k - 1 - support_num - 1 - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = generated[:, k - 1 - support_num - 1 - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label

    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    return images, x_total_images, y_total_image, y_few_shot_total_image


def sample_generator_store_into_file_for_quality(num_generations, sess, same_images, dropout_rate, dropout_rate_value,
                                                 data,
                                                 batch_size, file_name,
                                                 conditional_inputs, input_a,
                                                 support_input, input_b,
                                                 y_input_i, input_y_i,
                                                 y_input_j, input_y_j,
                                                 y_global_input_i, input_global_y_i,
                                                 y_global_input_j, input_global_y_j,
                                                 classes_number, classes,
                                                 selected_classes, classes_selected,
                                                 support_number, number_support,
                                                 z_input, z_input_2,
                                                 z_vectors, z_vectors_2,
                                                 training_phase, is_training,
                                                 z1z2_training, training_z1z2):
    input_images, support_images, generated, few_shot_y_label, y_label, similarities, f_encode_z, matching_feature = sess.run(
        same_images, feed_dict={
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: 1,
            z_input: batch_size * [z_vectors[0]],
            z_input_2: batch_size * [z_vectors_2[0]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]

    support_num = support_images.shape[-4]
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    recons_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                         input_images.shape[-1]))

    refer_trans_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                       input_images.shape[-1]))

    support_list = np.zeros(shape=(
        batch_size, num_generations, support_images.shape[-4], support_images.shape[-3], support_images.shape[-2],
        support_images.shape[-1]))
    #
    # similarities_list = np.zeros(shape=(batch_size, num_generations, support_num))
    # f_encode_z_list = np.zeros(shape=(batch_size, num_generations, 1024))
    # matching_feature_list = np.zeros(shape=(batch_size, num_generations, 1024))

    height = generated.shape[-3]

    is_interpolation = False
    num_interpolation = num_generations

    if is_interpolation:
        for i in range(num_interpolation):
            # z_vectors[i] = z_vectors[0]*(1-i*1e-1) + z_vectors[-1]*(i*1e-1)
            input_images, support_images, generated, few_shot_y_label, y_label, similarities, f_encode_z, matching_feature = sess.run(
                same_images, feed_dict={
                    input_a: conditional_inputs,
                    input_b: support_input,
                    input_y_i: y_input_i,
                    input_y_j: y_input_j,
                    input_global_y_i: y_global_input_i,
                    input_global_y_j: y_global_input_j,
                    classes: classes_number,
                    classes_selected: selected_classes,
                    number_support: support_number,
                    z_input: batch_size * [z_vectors[i]],
                    z_input_2: batch_size * [z_vectors_2[i]],
                    dropout_rate: dropout_rate_value,
                    training_phase: is_training,
                    z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            recons_images_list[:, i] = recons_image
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            # similarities_list[:, i] = similarities
            # f_encode_z_list[:, i] = f_encode_z
            # matching_feature_list[:,i] = matching_feature

    else:
        for i in range(num_generations):
            input_images, support_images, generated, few_shot_y_label, y_label, similarities, f_encode_z, matching_feature = sess.run(
                same_images, feed_dict={z_input: batch_size * [z_vectors[i]],
                                        input_a: conditional_inputs,
                                        input_b: support_input,
                                        input_y_i: y_input_i,
                                        input_y_j: y_input_j,
                                        input_global_y_i: y_global_input_i,
                                        input_global_y_j: y_global_input_j,
                                        classes: classes_number,
                                        classes_selected: selected_classes,
                                        number_support: support_number,
                                        z_input: batch_size * [z_vectors[i]],
                                        z_input_2: batch_size * [z_vectors_2[i]],
                                        dropout_rate: dropout_rate_value,
                                        training_phase: is_training,
                                        z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            recons_images_list[:, i] = recons_image
            refer_trans_list[:, i] = refer_trans

    input_images, support_images, generated, reconstructed, refer_translation = data.reconstruct_original(
        input_images_list), data.reconstruct_original(
        support_list), data.reconstruct_original(generated_list), data.reconstruct_original(
        recons_images_list), data.reconstruct_original(refer_trans_list)
    images = np.zeros(
        shape=(
            batch_size, 1 + support_num + 1 + 1 + num_generations, support_images.shape[-3], support_images.shape[-2],
            support_images.shape[-1]))

    # for i in range(num_generations):
    #     print('index of image',i)
    #     print('image difference',np.sum(generated[0]-generated[i]))

    x_total_images = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + num_generations), image_size, image_size, channel])
    y_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), classes_number])
    y_few_shot_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), selected_classes])

    for k in range(1 + support_num + 1 + 1 + num_generations):
        if k == 0:
            images[:, k] = input_images[:, 0]
            x_total_images[:batch_size] = input_images[:, 0]
            y_total_image[:batch_size] = y_label
            y_few_shot_total_image[:batch_size] = few_shot_y_label
        elif k > 0 and k < support_num + 1:
            images[:, k] = support_images[:, 0, k - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = support_images[:, 0, k - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 1:
            images[:, k] = reconstructed[:, 0]
            x_total_images[batch_size * k:batch_size * (k + 1)] = reconstructed[:, 0]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 2:
            images[:, k] = refer_translation[:, 0]
        else:
            images[:, k] = generated[:, k - 1 - support_num - 1 - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = generated[:, k - 1 - support_num - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label

    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    image_size = generated.shape[-3]
    print('hehreh', file_name)
    for j in range(batch_size):
        current_path = file_name.split('//')[0] + '/{}/'.format(np.argmax(y_global_input_i[:, j]))
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        for k in range(1 + support_number + 1 + 1 + num_generations):
            ##### generating images for calculating
            if k > 0 and k < support_num + 1:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_conditional_{}.png'.format(
                    k)
                scipy.misc.imsave(current_name, current_iamge)
            elif k == support_num + 1:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_cyclerecons_{}.png'.format(
                    k)
            elif k == support_num + 2:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_recons_{}.png'.format(k)
            else:
                if (find_classes(current_path) < num_generations):
                    current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                    if (current_iamge.shape[-2] < 128):
                        current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)

                    current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_sample_{}.png'.format(
                        k)
                    scipy.misc.imsave(current_name, current_iamge)

    scipy.misc.imsave(file_name, images)

    return x_total_images, y_total_image, y_few_shot_total_image


def sample_generator_store_into_file_for_classifier(num_generations, sess, same_images, dropout_rate,
                                                    dropout_rate_value, data,
                                                    batch_size, file_name,
                                                    conditional_inputs, input_a,
                                                    support_input, input_b,
                                                    y_input_i, input_y_i,
                                                    y_input_j, input_y_j,
                                                    y_global_input_i, input_global_y_i,
                                                    y_global_input_j, input_global_y_j,
                                                    classes_number, classes,
                                                    selected_classes, classes_selected,
                                                    support_number, number_support,
                                                    z_input, z_input_2,
                                                    z_vectors, z_vectors_2,
                                                    training_phase, is_training,
                                                    z1z2_training, training_z1z2):
    input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
        same_images, feed_dict={
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: 1,
            z_input: batch_size * [z_vectors[0]],
            z_input_2: batch_size * [z_vectors_2[0]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]

    support_num = support_images.shape[-4]
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    recons_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                         input_images.shape[-1]))

    refer_trans_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                       input_images.shape[-1]))

    support_list = np.zeros(shape=(
        batch_size, num_generations, support_images.shape[-4], support_images.shape[-3], support_images.shape[-2],
        support_images.shape[-1]))

    # similarities_list = np.zeros(shape=(batch_size, num_generations, support_num))
    # f_encode_z_list = np.zeros(shape=(batch_size, num_generations, 1024))
    # matching_feature_list = np.zeros(shape=(batch_size, num_generations, 1024))

    height = generated.shape[-3]

    is_interpolation = False
    num_interpolation = num_generations

    if is_interpolation:
        for i in range(num_interpolation):
            # z_vectors[i] = z_vectors[0]*(1-i*1e-1) + z_vectors[-1]*(i*1e-1)
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={
                    input_a: conditional_inputs,
                    input_b: support_input,
                    input_y_i: y_input_i,
                    input_y_j: y_input_j,
                    input_global_y_i: y_global_input_i,
                    input_global_y_j: y_global_input_j,
                    classes: classes_number,
                    classes_selected: selected_classes,
                    number_support: support_number,
                    z_input: batch_size * [z_vectors[i]],
                    z_input_2: batch_size * [z_vectors_2[i]],
                    dropout_rate: dropout_rate_value,
                    training_phase: is_training,
                    z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            recons_images_list[:, i] = recons_image
            generated_list[:, i] = generated
            support_list[:, i] = support_images

        # similarities_list[:, i] = similarities
        # f_encode_z_list[:, i] = f_encode_z
        # matching_feature_list[:,i] = matching_feature

    else:
        for i in range(num_generations):
            input_images, support_images, generated, recons_image, refer_trans, few_shot_y_label, y_label = sess.run(
                same_images, feed_dict={z_input: batch_size * [z_vectors[i]],
                                        input_a: conditional_inputs,
                                        input_b: support_input,
                                        input_y_i: y_input_i,
                                        input_y_j: y_input_j,
                                        input_global_y_i: y_global_input_i,
                                        input_global_y_j: y_global_input_j,
                                        classes: classes_number,
                                        classes_selected: selected_classes,
                                        number_support: support_number,
                                        z_input: batch_size * [z_vectors[i]],
                                        z_input_2: batch_size * [z_vectors_2[i]],
                                        dropout_rate: dropout_rate_value,
                                        training_phase: is_training,
                                        z1z2_training: training_z1z2})
            # print('1',type(input_images))
            input_images_list[:, i] = input_images
            generated_list[:, i] = generated
            support_list[:, i] = support_images
            recons_images_list[:, i] = recons_image
            refer_trans_list[:, i] = refer_trans
            # similarities_list[:, i] = similarities

    input_images, support_images, generated, reconstructed, refer_translation = data.reconstruct_original(
        input_images_list), data.reconstruct_original(
        support_list), data.reconstruct_original(generated_list), data.reconstruct_original(
        recons_images_list), data.reconstruct_original(refer_trans_list)
    images = np.zeros(
        shape=(
            batch_size, 1 + support_num + 1 + 1 + num_generations, support_images.shape[-3], support_images.shape[-2],
            support_images.shape[-1]))

    x_total_images = np.zeros(
        [batch_size * (1 + support_num + 1 + 1 + num_generations), image_size, image_size, channel])
    y_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), classes_number])
    y_few_shot_total_image = np.zeros([batch_size * (1 + support_num + 1 + 1 + num_generations), selected_classes])

    for k in range(1 + support_num + 1 + 1 + num_generations):
        if k == 0:
            images[:, k] = input_images[:, 0]
            x_total_images[:batch_size] = input_images[:, 0]
            y_total_image[:batch_size] = y_label
            y_few_shot_total_image[:batch_size] = few_shot_y_label
        elif k > 0 and k < support_num + 1:
            images[:, k] = support_images[:, 0, k - 1]
            x_total_images[batch_size * k:batch_size * (k + 1)] = support_images[:, 0, k - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 1:
            images[:, k] = reconstructed[:, 0]
            x_total_images[batch_size * k:batch_size * (k + 1)] = reconstructed[:, 0]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label
        elif k == support_num + 2:
            images[:, k] = refer_translation[:, 0]
        else:
            # print('here_{}'.format(k))
            images[:, k] = generated[:, k - 1 - support_num - 1 - 1]
            # print('match_{}'.format(k - 1 - support_num -1 -1 ))
            x_total_images[batch_size * k:batch_size * (k + 1)] = generated[:, k - 1 - support_num - 1 - 1]
            y_total_image[batch_size * k:batch_size * (k + 1)] = y_label

    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)

    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    image_size = generated.shape[-3]
    for j in range(batch_size):
        current_path = file_name.split('//')[0] + '/{}/'.format(np.argmax(y_global_input_i[:, j]))
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        for k in range(1 + support_number + 1 + 1 + num_generations):
            ##### generating images for calculating
            if k >= 0 and k < support_num + 1:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_conditional_{}.png'.format(
                    k)
                scipy.misc.imsave(current_name, current_iamge)
            elif k == support_num + 1:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_cyclerecons_{}.png'.format(
                    k)
                scipy.misc.imsave(current_name, current_iamge)
            elif k == support_num + 2:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)
                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_recons_{}.png'.format(k)
                scipy.misc.imsave(current_name, current_iamge)
            else:
                current_iamge = images[image_size * j:image_size * (j + 1), image_size * (k):image_size * (k + 1)]
                if (current_iamge.shape[-2] < 128):
                    current_iamge = cv2.resize(current_iamge, (128, 128), interpolation=cv2.INTER_LINEAR)

                current_name = current_path + file_name.split('//')[-1].split('.png')[0] + '_sample_{}.png'.format(
                    k)
                scipy.misc.imsave(current_name, current_iamge)

    return x_total_images, y_total_image, y_few_shot_total_image


def sample_generator_for_classifier(num_generations, iteration, sess, same_images,
                                    dropout_rate, dropout_rate_value, data, batch_size, file_name,
                                    conditional_inputs, input_a,
                                    support_input, input_b,
                                    y_input_i, input_y_i,
                                    y_input_j, input_y_j,
                                    y_global_input_i, input_global_y_i,
                                    y_global_input_j, input_global_y_j,
                                    input_global_x_j_selected, selected_global_x_j,
                                    input_global_y_j_selected, selected_global_y_j,
                                    classes_number, classes,
                                    selected_classes, classes_selected,
                                    support_number, number_support,
                                    z_input, z_input_2,
                                    z_vectors, z_vectors_2,
                                    training_phase, is_training,
                                    z1z2_training, training_z1z2, z_dim,
                                    feed_augmented, augmented_number,
                                    feed_confidence, confidence,
                                    feed_loss_d, loss_d):
    support_input_selected = support_input[:, :, :support_number]
    y_input_j_selected = y_input_j[:, :, :support_number]
    y_global_input_j_selected = y_global_input_j[:, :, :support_number]

    input_images, support_images, generated, few_shot_y_batch, y_batch_r, few_shot_y_support, y_support, similarities, \
    d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
        input_a: conditional_inputs,
        input_b: support_input_selected,
        input_y_i: y_input_i,
        input_y_j: y_input_j_selected,
        input_global_y_i: y_global_input_i,
        input_global_y_j: y_global_input_j_selected,
        # selected_global_x_j:input_global_x_j_selected,
        # selected_global_y_j:input_global_y_j_selected,
        classes: classes_number,
        classes_selected: data.selected_classes,
        number_support: data.support_number,
        z_input: batch_size * [z_vectors[0]],
        z_input_2: batch_size * [z_vectors_2[0]],
        dropout_rate: dropout_rate_value,
        training_phase: is_training,
        z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]
    height = generated.shape[-3]
    time_1 = time.time()

    generated_images = []
    generated_labels = []
    generated_fewshot_labels = []

    augmented_support_image = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, height, height, channel])
    augmented_few_shot_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, data.selected_classes])
    augmented_global_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, classes_number])

    # print(np.max(support_input),np.min(support_input))
    augmented_support_image[:, :data.support_number * data.selected_classes] = support_input[0]
    augmented_few_shot_label[:, :data.support_number * data.selected_classes] = y_input_j[0]
    augmented_global_label[:, :data.support_number * data.selected_classes] = y_global_input_j[0]

    images = np.zeros([batch_size, augmented_number * data.selected_classes, height, height, channel])
    few_shot_labels = np.zeros([batch_size, augmented_number * data.selected_classes, data.selected_classes])
    global_labels = np.zeros([batch_size, augmented_number * data.selected_classes, classes_number])

    for j in range(data.selected_classes):
        number_flags = np.zeros([batch_size])
        x_batch = support_input[:, :, data.support_number * j, :, :, :]
        y_batch = y_input_j[:, :, data.support_number * j, :]
        y_batch_global = y_global_input_j[:, :, data.support_number * j, :]

        support_input_selected = support_input[:, :, j * data.support_number:j * data.support_number + support_number,
                                 :, :, :]
        y_input_j_selected = y_input_j[:, :, j * data.support_number:j * data.support_number + support_number, :]
        y_global_input_j_selected = y_global_input_j[:, :,
                                    j * data.support_number:j * data.support_number + support_number, :]

        for i in range(augmented_number):
            z_vectors_current = np.random.normal(size=(10, z_dim))
            z_vectors_2_current = np.random.normal(size=(10, z_dim))
            input_images, support_images, generated, few_shot_y_batch, y_batch_r, few_shot_y_support, y_support, similarities, \
            d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
                z_input: batch_size * [z_vectors_current[0]],
                input_a: x_batch,
                input_b: support_input_selected,
                input_y_i: y_batch,
                input_y_j: y_input_j_selected,
                input_global_y_i: y_batch_global,
                input_global_y_j: y_global_input_j_selected,
                # selected_global_x_j:input_global_x_j_selected,
                # selected_global_y_j:input_global_y_j_selected,
                classes: classes_number,
                classes_selected: data.selected_classes,
                number_support: data.support_number,
                z_input: batch_size * [z_vectors_current[0]],
                z_input_2: batch_size * [z_vectors_2_current[0]],
                dropout_rate: dropout_rate_value,
                training_phase: is_training,
                z1z2_training: training_z1z2})

            augmented_support_image[:, data.support_number * data.selected_classes + i] = generated
            augmented_few_shot_label[:, data.support_number * data.selected_classes + i] = few_shot_y_batch
            augmented_global_label[:, data.support_number * data.selected_classes + i] = y_batch_r

    save = False
    if save:
        if iteration == 0:
            file_num = int(augmented_number / 32)
            for j in range(data.selected_classes):
                for s in range(file_num):
                    current_images = unstack(augmented_support_image[:,
                                             augmented_number * j + 32 * s + data.support_number * data.selected_classes: augmented_number * j + 32 * (
                                                     s + 1) + data.support_number * data.selected_classes])
                    current_images = np.concatenate((current_images), axis=1)
                    current_images = unstack(current_images)
                    current_images = np.concatenate((current_images), axis=1)
                    current_images = np.squeeze(current_images)
                    current_images = data.reconstruct_original(current_images)
                    current_images = (current_images - np.min(current_images)) / (
                            np.max(current_images) - np.min(current_images))
                    current_images = current_images * 255

                    current_file_name = file_name.split('png')[0] + '{}_{}.png'.format(j, s)
                    print('storing image', current_file_name)
                    scipy.misc.imsave(current_file_name, current_images)

    return augmented_support_image, augmented_few_shot_label, augmented_global_label


def sample_generator_selector(num_generations, sess, same_images,
                              dropout_rate, dropout_rate_value, data, batch_size, file_name,
                              conditional_inputs, input_a,
                              support_input, input_b,
                              y_input_i, input_y_i,
                              y_input_j, input_y_j,
                              y_global_input_i, input_global_y_i,
                              y_global_input_j, input_global_y_j,
                              input_global_x_j_selected, selected_global_x_j,
                              input_global_y_j_selected, selected_global_y_j,
                              classes_number, classes,
                              selected_classes, classes_selected,
                              support_number, number_support,
                              z_input, z_input_2,
                              z_vectors, z_vectors_2,
                              training_phase, is_training,
                              z1z2_training, training_z1z2,
                              feed_augmented, augmented_number,
                              feed_confidence, confidence,
                              feed_loss_d, loss_d):
    is_saved = True
    input_images, support_images, generated, few_shot_y_batch, y_batch, few_shot_y_support, y_support, similarities, \
    d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
        input_a: conditional_inputs,
        input_b: support_input,
        input_y_i: y_input_i,
        input_y_j: y_input_j,
        input_global_y_i: y_global_input_i,
        input_global_y_j: y_global_input_j,
        selected_global_x_j: input_global_x_j_selected,
        selected_global_y_j: input_global_y_j_selected,
        classes: classes_number,
        classes_selected: selected_classes,
        number_support: support_number,
        z_input: batch_size * [z_vectors[0]],
        z_input_2: batch_size * [z_vectors_2[0]],
        dropout_rate: dropout_rate_value,
        training_phase: is_training,
        z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]
    height = generated.shape[-3]

    time_1 = time.time()
    for i in range(num_generations):
        input_images, support_images, generated, few_shot_y_batch, y_batch, few_shot_y_support, y_support, similarities, \
        d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
            z_input: batch_size * [z_vectors[i]],
            input_a: conditional_inputs,
            input_b: support_input,
            input_y_i: y_input_i,
            input_y_j: y_input_j,
            input_global_y_i: y_global_input_i,
            input_global_y_j: y_global_input_j,
            selected_global_x_j: input_global_x_j_selected,
            selected_global_y_j: input_global_y_j_selected,
            classes: classes_number,
            classes_selected: selected_classes,
            number_support: support_number,
            z_input: batch_size * [z_vectors[i]],
            z_input_2: batch_size * [z_vectors_2[i]],
            dropout_rate: dropout_rate_value,
            training_phase: is_training,
            z1z2_training: training_z1z2})

        few_shot_fake_category = np.expand_dims(few_shot_fake_category, -1)
        few_shot_confidence_score = np.expand_dims(few_shot_confidence_score, -1)
        generated = np.expand_dims(generated, 1)
        if i == 0:
            generated_images = generated
            generated_confidence = few_shot_confidence_score
            generated_loss = d_loss
        else:
            generated_images = np.hstack((generated_images, generated))
            generated_confidence = np.hstack((generated_confidence, few_shot_confidence_score))
            generated_loss = np.hstack((generated_loss, d_loss))
    time_2 = time.time()
    print('time generating images:', time_2 - time_1)

    if is_saved:
        images = np.zeros(shape=(
            batch_size, 1 + selected_classes * (support_number + augmented_number), support_images.shape[-3],
            support_images.shape[-2],
            support_images.shape[-1]))
        input_images = data.reconstruct_original(input_images)
        support_images = data.reconstruct_original(support_images)
        generated_images = data.reconstruct_original(generated_images)
        images[:, 0] = input_images
        images[:, 1:1 + selected_classes * support_number] = support_images
    else:
        images = np.zeros(shape=(
            batch_size, selected_classes * (support_number + augmented_number), support_images.shape[-3],
            support_images.shape[-2],
            support_images.shape[-1]))

    for k in range(batch_size):
        for i in range(selected_classes):
            index_current = np.where(generated_fake_categores[k] == i)[0]
            current_generated_images = generated_images[k, index_current, :, :, :]
            current_confidence = generated_confidence[k, index_current]
            current_loss = generated_loss[k, index_current]
            length = len(index_current)
            ratio = augmented_number / length
            current_category_images = []
            current_category_fewshot_label[k, :, i] = 1
            current_category_global_label[k, :, fewshot_global[i]] = 1
            if confidence > 0 and loss_d > 0:
                fusion = current_confidence * current_loss
                fusion_selection = np.sort(fusion, axis=0)
                fusion_selection_threshod = fusion_selection[int(length * (1 - ratio)) - 1]
                for j in range(length):
                    if fusion[j] >= fusion_selection_threshod:
                        current_category_images.append(current_generated_images[j])
            elif confidence > 0:
                confidence = np.sort(current_confidence, axis=0)
                confidence_threshod = confidence[int(length * (1 - ratio)) - 1]
                for j in range(length):
                    if current_confidence[j] >= confidence_threshod:
                        current_category_images.append(current_generated_images[j])

            elif loss_d > 0:
                loss_d = np.sort(current_loss, axis=0)
                d_loss_threshod = loss_d[int(length * (1 - ratio)) - 1]
                for j in range(length):
                    if current_loss[j] >= d_loss_threshod:
                        current_category_images.append(current_generated_images[j])

            images[k,
            1 + selected_classes * support_number + i * augmented_number:1 + selected_classes * support_number + (
                    i + 1) * augmented_number] = data.reconstruct_original(
                np.array(current_category_images[:augmented_number]))
            few_shot_label[k,
            1 + selected_classes * support_number + i * augmented_number:1 + selected_classes * support_number + (
                    i + 1) * augmented_number] = current_category_fewshot_label[k]
            global_label[k,
            1 + selected_classes * support_number + i * augmented_number:1 + selected_classes * support_number + (
                    i + 1) * augmented_number] = current_category_global_label[k]

    time_3 = time.time()

    global_images = images
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = unstack(images)
    images = np.concatenate((images), axis=1)
    images = np.squeeze(images)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    images = images * 255

    time_4 = time.time()
    # print('time for selecting images',time_4 - time_3)

    scipy.misc.imsave(file_name, images)
    time_5 = time.time()
    return global_images[:, 1:, :, :, :], few_shot_label[:, 1:, :], global_label[:, 1:, :]


def sample_generator_selector_on_similarity(num_generations, iteration, sess, same_images,
                                            dropout_rate, dropout_rate_value, data, batch_size, file_name,
                                            conditional_inputs, input_a,
                                            support_input, input_b,
                                            y_input_i, input_y_i,
                                            y_input_j, input_y_j,
                                            y_global_input_i, input_global_y_i,
                                            y_global_input_j, input_global_y_j,
                                            input_global_x_j_selected, selected_global_x_j,
                                            input_global_y_j_selected, selected_global_y_j,
                                            classes_number, classes,
                                            selected_classes, classes_selected,
                                            support_number, number_support,
                                            z_input, z_input_2,
                                            z_vectors, z_vectors_2,
                                            training_phase, is_training,
                                            z1z2_training, training_z1z2, z_dim,
                                            feed_augmented, augmented_number,
                                            feed_confidence, confidence,
                                            feed_loss_d, loss_d):
    #### data: support and selected classes: data.support_number, data.selected_classes
    #### matchingGAN: support and selected classes: selected_classes, support_number

    ### feed_dict can keep consist with the mathcingGAN
    ### the input data should keep consistent with the data generation, howe to conduct

    ##### such input images all from the data setting(data.selected_classes, data.support_number)

    support_input_selected = support_input[:, :, :support_number]
    y_input_j_selected = y_input_j[:, :, :support_number]
    y_global_input_j_selected = y_global_input_j[:, :, :support_number]

    # print('1',np.shape(support_input_selected))

    input_images, support_images, generated, few_shot_y_batch, y_batch_r, few_shot_y_support, y_support, similarities, \
    d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
        input_a: conditional_inputs,
        input_b: support_input_selected,
        input_y_i: y_input_i,
        input_y_j: y_input_j_selected,
        input_global_y_i: y_global_input_i,
        input_global_y_j: y_global_input_j_selected,
        selected_global_x_j: input_global_x_j_selected,
        selected_global_y_j: input_global_y_j_selected,
        classes: classes_number,
        classes_selected: data.selected_classes,
        number_support: data.support_number,
        z_input: batch_size * [z_vectors[0]],
        z_input_2: batch_size * [z_vectors_2[0]],
        dropout_rate: dropout_rate_value,
        training_phase: is_training,
        z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]
    height = generated.shape[-3]
    time_1 = time.time()

    generated_images = []
    generated_labels = []
    generated_fewshot_labels = []

    augmented_support_image = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, height, height, channel])
    augmented_few_shot_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, data.selected_classes])
    augmented_global_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, classes_number])

    # print(np.max(support_input),np.min(support_input))
    augmented_support_image[:, :data.support_number * data.selected_classes] = support_input[0]
    augmented_few_shot_label[:, :data.support_number * data.selected_classes] = y_input_j[0]
    augmented_global_label[:, :data.support_number * data.selected_classes] = y_global_input_j[0]

    images = np.zeros([batch_size, augmented_number * data.selected_classes, height, height, channel])
    few_shot_labels = np.zeros([batch_size, augmented_number * data.selected_classes, data.selected_classes])
    global_labels = np.zeros([batch_size, augmented_number * data.selected_classes, classes_number])

    for j in range(data.selected_classes):
        print('current class:', j)
        number_flags = np.zeros([batch_size])
        x_batch = support_input[:, :, data.support_number * j, :, :, :]
        y_batch = y_input_j[:, :, data.support_number * j, :]
        y_batch_global = y_global_input_j[:, :, data.support_number * j, :]

        support_input_selected = support_input[:, :, j * data.support_number:j * data.support_number + support_number,
                                 :, :, :]
        y_input_j_selected = y_input_j[:, :, j * data.support_number:j * data.support_number + support_number, :]
        y_global_input_j_selected = y_global_input_j[:, :,
                                    j * data.support_number:j * data.support_number + support_number, :]

        for i in range(num_generations):
            z_vectors_current = np.random.normal(size=(10, z_dim))
            z_vectors_2_current = np.random.normal(size=(10, z_dim))
            input_images, support_images, generated, few_shot_y_batch, y_batch_r, few_shot_y_support, y_support, similarities, \
            d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
                z_input: batch_size * [z_vectors_current[0]],
                input_a: x_batch,
                input_b: support_input_selected,
                input_y_i: y_batch,
                input_y_j: y_input_j_selected,
                input_global_y_i: y_batch_global,
                input_global_y_j: y_global_input_j_selected,
                selected_global_x_j: input_global_x_j_selected,
                selected_global_y_j: input_global_y_j_selected,
                classes: classes_number,
                classes_selected: data.selected_classes,
                number_support: data.support_number,
                z_input: batch_size * [z_vectors_current[0]],
                z_input_2: batch_size * [z_vectors_2_current[0]],
                dropout_rate: dropout_rate_value,
                training_phase: is_training,
                z1z2_training: training_z1z2})

            similarities_threshold = np.min(similarities, axis=1)
            for k in range(batch_size):
                if similarities_threshold[k] > 0:
                    if number_flags[k] < augmented_number:
                        images[k, j * augmented_number + int(number_flags[k]), :, :, :] = generated[k, :, :, :]
                        few_shot_labels[k, j * augmented_number + int(number_flags[k]), :] = few_shot_y_batch[k, :]
                        global_labels[k, j * augmented_number + int(number_flags[k]), :] = y_batch_r[k, :]
                        number_flags[k] += 1
            if np.min(number_flags) == augmented_number:
                print('current class image generation finished')
                break

    print('finished all images generation')
    augmented_support_image[:, data.support_number * data.selected_classes:] = images
    augmented_few_shot_label[:, data.support_number * data.selected_classes:] = few_shot_labels
    augmented_global_label[:, data.support_number * data.selected_classes:] = global_labels
    if iteration == 0:
        file_num = int(augmented_number / 32)
        for j in range(data.selected_classes):
            for s in range(file_num):
                current_images = unstack(images[:, augmented_number * j + 32 * s: augmented_number * j + 32 * (s + 1)])
                current_images = np.concatenate((current_images), axis=1)
                current_images = unstack(current_images)
                current_images = np.concatenate((current_images), axis=1)
                current_images = np.squeeze(current_images)
                # images = (images - np.min(images)) / (np.max(images) - np.min(images))
                # images = images * 255
                current_images = data.reconstruct_original(current_images)
                current_images = (current_images - np.min(current_images)) / (
                        np.max(current_images) - np.min(current_images))
                current_images = current_images * 255
                current_file_name = file_name.split('.')[0] + '{}_{}.png'.format(j, s)
                scipy.misc.imsave(current_file_name, current_images)

        real_images = support_input[0]
        real_images = np.concatenate((real_images), axis=1)
        real_images = unstack(real_images)
        real_images = np.concatenate((real_images), axis=1)
        real_images = np.squeeze(real_images)
        # images = (images - np.min(images)) / (np.max(images) - np.min(images))
        # images = images * 255
        real_images = data.reconstruct_original(real_images)
        real_images = (real_images - np.min(real_images)) / (np.max(real_images) - np.min(real_images))
        real_images = real_images * 255
        current_file_name = file_name.split('.')[0] + 'real_{}.png'.format(j)
        scipy.misc.imsave(file_name, real_images)

    return augmented_support_image, augmented_few_shot_label, augmented_global_label


def sample_generator_selector_on_batch(num_generations, iteration, sess, same_images,
                                       dropout_rate, dropout_rate_value, data, batch_size, file_name,
                                       conditional_inputs, input_a,
                                       support_input, input_b,
                                       y_input_i, input_y_i,
                                       y_input_j, input_y_j,
                                       y_global_input_i, input_global_y_i,
                                       y_global_input_j, input_global_y_j,
                                       input_global_x_j_selected, selected_global_x_j,
                                       input_global_y_j_selected, selected_global_y_j,
                                       classes_number, classes,
                                       selected_classes, classes_selected,
                                       support_number, number_support,
                                       z_input, z_input_2,
                                       z_vectors, z_vectors_2,
                                       training_phase, is_training,
                                       z1z2_training, training_z1z2,
                                       feed_augmented, augmented_number,
                                       feed_confidence, confidence,
                                       feed_loss_d, loss_d):
    #### data: support and selected classes: data.support_number, data.selected_classes
    #### matchingGAN: support and selected classes: selected_classes, support_number

    ### feed_dict can keep consist with the mathcingGAN
    ### the input data should keep consistent with the data generation, howe to conduct

    ##### such input images all from the data setting(data.selected_classes, data.support_number)

    support_input_selected = support_input[:, :, :support_number]
    y_input_j_selected = y_input_j[:, :, :support_number]
    y_global_input_j_selected = y_global_input_j[:, :, :support_number]

    # print('1',np.shape(support_input_selected))

    input_images, support_images, generated, few_shot_y_batch, y_batch, few_shot_y_support, y_support, similarities, \
    d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
        input_a: conditional_inputs,
        input_b: support_input_selected,
        input_y_i: y_input_i,
        input_y_j: y_input_j_selected,
        input_global_y_i: y_global_input_i,
        input_global_y_j: y_global_input_j_selected,
        selected_global_x_j: input_global_x_j_selected,
        selected_global_y_j: input_global_y_j_selected,
        classes: classes_number,
        classes_selected: data.selected_classes,
        number_support: data.support_number,
        z_input: batch_size * [z_vectors[0]],
        z_input_2: batch_size * [z_vectors_2[0]],
        dropout_rate: dropout_rate_value,
        training_phase: is_training,
        z1z2_training: training_z1z2})

    image_size = input_images.shape[1]
    channel = input_images.shape[-1]
    height = generated.shape[-3]
    time_1 = time.time()

    #### genrate image and labels for each batch image
    generated_images = np.zeros([batch_size, data.selected_classes, num_generations, height, height, channel])
    selected_images = np.zeros([batch_size, data.selected_classes, augmented_number, height, height, channel])
    generated_confidence = np.zeros([batch_size, data.selected_classes, num_generations])
    generated_loss = np.zeros([batch_size, data.selected_classes, num_generations])
    generated_global_label = np.zeros([batch_size, data.selected_classes, num_generations, y_support.shape[-1]])
    selected_global_label = np.zeros([batch_size, data.selected_classes, augmented_number, y_support.shape[-1]])
    generated_fewshot_label = np.zeros(
        [batch_size, data.selected_classes, num_generations, few_shot_y_support.shape[-1]])
    selected_fewshot_label = np.zeros(
        [batch_size, data.selected_classes, augmented_number, few_shot_y_support.shape[-1]])
    #### genrate image and labels for each batch image

    for i in range(num_generations):
        for j in range(data.selected_classes):
            x_batch = support_input[:, :, data.support_number * j, :, :, :]
            y_batch = y_input_j[:, :, data.support_number * j, :]
            y_batch_global = y_global_input_j[:, :, data.support_number * j, :]

            support_input_selected = support_input[:, :,
                                     j * data.support_number:j * data.support_number + support_number, :, :, :]
            y_input_j_selected = y_input_j[:, :, j * data.support_number:j * data.support_number + support_number, :]
            y_global_input_j_selected = y_global_input_j[:, :,
                                        j * data.support_number:j * data.support_number + support_number, :]

            # print('2',np.shape(support_input_selected))

            input_images, support_images, generated, few_shot_y_batch, y_batch, few_shot_y_support, y_support, similarities, \
            d_loss, few_shot_fake_category, few_shot_confidence_score = sess.run(same_images, feed_dict={
                z_input: batch_size * [z_vectors[i]],
                input_a: x_batch,
                input_b: support_input_selected,
                input_y_i: y_batch,
                input_y_j: y_input_j_selected,
                input_global_y_i: y_batch_global,
                input_global_y_j: y_global_input_j_selected,
                selected_global_x_j: input_global_x_j_selected,
                selected_global_y_j: input_global_y_j_selected,
                classes: classes_number,
                classes_selected: data.selected_classes,
                number_support: data.support_number,
                z_input: batch_size * [z_vectors[i]],
                z_input_2: batch_size * [z_vectors_2[i]],
                dropout_rate: dropout_rate_value,
                training_phase: is_training,
                z1z2_training: training_z1z2})

            generated_images[:, j, i, :, :, :] = generated
            generated_global_label[:, j, i, :] = y_batch
            generated_fewshot_label[:, j, i, :] = few_shot_y_batch
            generated_confidence[:, j, i] = few_shot_confidence_score
            generated_loss[:, j, i] = np.squeeze(d_loss, axis=-1)
            # print(d_loss)
            # print(few_shot_confidence_score)
    time_2 = time.time()

    # input_images = data.reconstruct_original(input_images)
    # support_images = data.reconstruct_original(support_images)
    ratio = augmented_number / num_generations
    for i in range(data.selected_classes):
        if confidence > 0 and loss_d > 0:
            fusion = generated_confidence[:, i] * generated_loss[:, i]
            fusion_selection = np.sort(fusion, axis=1)
            fusion_selection_threshod = fusion_selection[:, int(num_generations * (1 - ratio)) - 1]
            fusion_selection_threshod_tile = (np.tile(fusion_selection_threshod, (num_generations, 1))).T
            mask = np.where(fusion > fusion_selection_threshod_tile, 1, 0)
            # print(np.shape(mask)) (32, 20)
            mask_index = np.argwhere(mask > 0)

        elif confidence > 0:
            confidence = np.sort(generated_confidence[:, i], axis=1)
            confidence_threshod = confidence[:, int(num_generations * (1 - ratio)) - 1]
            confidence_threshod_tile = (np.tile(confidence_threshod, (num_generations, 1))).T
            mask = np.where(generated_confidence[:, i] > confidence_threshod_tile, 1, 0)
            mask_index = np.argwhere(mask > 0)

        elif loss_d > 0:
            loss_d_current = np.sort(generated_loss[:, i], axis=1)
            d_loss_threshod = loss_d_current[:, int(num_generations * (1 - ratio)) - 1]
            d_loss_threshod_tile = (np.tile(d_loss_threshod, (num_generations, 1))).T
            mask = np.where(generated_loss[:, i] > d_loss_threshod_tile, 1, 0)
            mask_index = np.argwhere(mask > 0)

        # print(np.shape(mask_index))  (320, 2) order by the batchsize
        # print(np.shape(generated_images[:,i])) (320,84,84,3)
        # print('here',mask_index)
        selected_images[:, i] = np.reshape(generated_images[:, i][mask_index[:, 0], mask_index[:, 1]],
                                           (batch_size, augmented_number, height, height, channel))
        selected_fewshot_label[:, i] = np.reshape(generated_fewshot_label[:, i][mask_index[:, 0], mask_index[:, 1]],
                                                  (batch_size, augmented_number, data.selected_classes))
        selected_global_label[:, i] = np.reshape(generated_global_label[:, i][mask_index[:, 0], mask_index[:, 1]],
                                                 (batch_size, augmented_number, classes_number))

        if iteration == 0:
            current_augmented_image = np.zeros(
                [batch_size, data.support_number + augmented_number, height, height, channel])
            current_augmented_image[:, :data.support_number, :, :, :] = support_input[:, :, j * data.support_number:(
                                                                                                                            j + 1) * data.support_number,
                                                                        :, :, :]
            current_augmented_image[:, data.support_number:, :, :, :] = selected_images[:, i]
            images = current_augmented_image
            images = unstack(images)
            images = np.concatenate((images), axis=1)
            images = unstack(images)
            images = np.concatenate((images), axis=1)
            images = np.squeeze(images)
            # images = (images - np.min(images)) / (np.max(images) - np.min(images))
            # images = images * 255
            images = data.reconstruct_original(images)
            images = (images - np.min(images)) / (np.max(images) - np.min(images))
            images = images * 255
            current_file_name = file_name.split('.')[0] + '{}.png'.format(i)
            scipy.misc.imsave(current_file_name, images)
            is_saved = False

    augmented_support_image = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, height, height, channel])
    augmented_few_shot_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, data.selected_classes])
    augmented_global_label = np.zeros(
        [batch_size, (augmented_number + data.support_number) * data.selected_classes, classes_number])

    # print(np.max(support_input),np.min(support_input))
    augmented_support_image[:, :data.support_number * data.selected_classes] = support_input
    augmented_few_shot_label[:, :data.support_number * data.selected_classes] = y_input_j
    augmented_global_label[:, :data.support_number * data.selected_classes] = y_global_input_j

    #####augmented data: images and labels
    augmented_support_image[:, data.support_number * data.selected_classes:] = np.reshape(selected_images, (
        batch_size, selected_images.shape[1] * selected_images.shape[2], height, height, channel))
    augmented_few_shot_label[:, data.support_number * data.selected_classes:] = np.reshape(selected_fewshot_label, (
        batch_size, selected_fewshot_label.shape[1] * selected_fewshot_label.shape[2], data.selected_classes))
    augmented_global_label[:, data.support_number * data.selected_classes:] = np.reshape(selected_global_label, (
        batch_size, selected_global_label.shape[1] * selected_global_label.shape[2], classes_number))

    return augmented_support_image, augmented_few_shot_label, augmented_global_label


### Interpolated spherical subspace
def sample_two_dimensions_generator(sess, same_images,
                                    dropout_rate, dropout_rate_value, data,
                                    batch_size, file_name, conditional_inputs, input_a, support_input, input_b, input_y,
                                    y_input, classes_number, classes,
                                    training_phase, z_input, z_input_2, z_vectors, z_vectors_2):
    num_generations = z_vectors.shape[0]
    row_num_generations = int(np.sqrt(num_generations))
    column_num_generations = int(np.sqrt(num_generations))

    input_images, support_images, generated, y_label, similarities = sess.run(same_images,
                                                                              feed_dict={input_a: conditional_inputs,
                                                                                         input_b: support_input,
                                                                                         input_y: y_input,
                                                                                         classes: classes_number,
                                                                                         dropout_rate: dropout_rate_value,
                                                                                         training_phase: False,
                                                                                         z_input: batch_size * [
                                                                                             z_vectors[0]],
                                                                                         z_input_2: batch_size * [
                                                                                             z_vectors_2[0]]})

    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    support_list = np.zeros(shape=(
        batch_size, num_generations, support_images.shape[-4], support_images.shape[-3], support_images.shape[-2],
        support_images.shape[-1]))

    height = generated.shape[-3]

    for i in range(num_generations):
        input_images, support_images, generated, y_label, similarities = sess.run(same_images, feed_dict={
            z_input: batch_size * [z_vectors[i]],
            z_input_2: batch_size * [z_vectors_2[i]],
            input_a: conditional_inputs,
            input_b: support_input,
            input_y: y_input,
            classes: classes_number,
            training_phase: False, dropout_rate:
                dropout_rate_value})
        input_images_list[:, i] = input_images
        generated_list[:, i] = generated

    input_images, generated = data.reconstruct_original(input_images_list), data.reconstruct_original(generated_list)
    im_size = generated.shape

    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    line = np.zeros(shape=(batch_size, 1, generated.shape[-3], generated.shape[-2],
                           generated.shape[-1]))

    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)

    image = np.concatenate((input_images, generated), axis=1)
    im_dimension = im_size[3]
    image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    full_image = image[:, (num_generations - 1) * height:]

    for i in range(batch_size):
        image = full_image[i * im_dimension:(i + 1) * im_dimension]
        seed_image = image[0:im_dimension, 0:im_dimension]
        gen_images = image[0:im_dimension, 2 * im_dimension:]
        image = np.concatenate((seed_image, gen_images), axis=1)

        properly_positioned_image = []
        for j in range(row_num_generations):
            start = im_dimension * j * row_num_generations
            stop = im_dimension * (j + 1) * row_num_generations

            row_image = image[:, start:stop]

            properly_positioned_image.append(row_image)

        positioned_image = np.concatenate(properly_positioned_image, axis=0)

        scipy.misc.imsave("{}_{}.png".format(file_name, i), positioned_image)



