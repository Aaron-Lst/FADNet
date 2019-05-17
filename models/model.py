from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
import util.vgg_simple as vgg


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.scale = 1
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.data_list = open(args.datalist, 'rt').read().splitlines()
        self.data_list = list(map(lambda x: x.split(' '), self.data_list))
        random.shuffle(self.data_list)
        self.model_flag = self.args.model_flag
        self.exp_num = self.args.exp_num
        self.train_dir = os.path.join(
            './checkpoints', self.model_flag + '_' + self.exp_num)
        self.restore_dir = os.path.join(
            './checkpoints', 'refine_6')
        self.restore_step = 132000
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = (len(self.data_list)) // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate
        self.load_step = args.load_step
        self.loss_thread = 0.004
        self.min_loss_val = 1
        self.epoch_loss = 0

    def input_producer(self, batch_size=10):
        def read_data():
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['/home/opt603/lst/data/GOPRO_Large/train/', self.data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['/home/opt603/lst/data/GOPRO_Large/train/', self.data_queue[1]])),
                                          channels=3)
            img_c = tf.image.decode_image(tf.read_file(tf.string_join(['/home/opt603/lst/data/GOPRO_Large/train/', self.data_queue[2]])),
                                          channels=3)
            img_a, img_b, img_c = preprocessing([img_a, img_b, img_c])

            img_c = tf.image.rgb_to_grayscale(img_c)

            return img_a, img_b, img_c

        def preprocessing(imgs):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model is not 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            img_crop = tf.unstack(tf.random_crop(tf.stack(imgs, axis=0), [3, self.crop_size, self.crop_size, self.chns]),
                                  axis=0)
            return img_crop

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]
            ed_list = List_all[:, 2]

            self.data_queue = tf.train.slice_input_producer(
                [in_list, gt_list, ed_list], capacity=20)
            image_in, image_gt, image_ed = read_data()
            batch_in, batch_gt, batch_ed = tf.train.batch(
                [image_in, image_gt, image_ed], batch_size=batch_size, num_threads=8, capacity=20)

        return batch_in, batch_gt, batch_ed

    def generator(self, inputs, reuse=False, scope='refine'):
        with tf.variable_scope(scope, reuse=reuse) as vs:
            pred, refine, ed = self.model_refine(inputs, scope)
            variables = tf.contrib.framework.get_variables(vs)
            return pred, refine, ed, variables

    def model_refine(self, inputs, name):
        refine = []
        bn_params = batch_norm_params()
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(
                                uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            n, h, w, c = inputs.get_shape().as_list()
            inp_pred = inputs
            scale = self.scale
            hi = int(round(h * scale))
            wi = int(round(w * scale))
            inp_blur = tf.image.resize_images(
                inputs, [hi, wi], method=0)
            inp_pred = tf.stop_gradient(
                tf.image.resize_images(inp_pred, [hi, wi], method=0))
            inp_all = tf.concat(
                [inp_blur, inp_pred], axis=3, name='inp')
            # encoder
            conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
            conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
            conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
            conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
            conv2_1 = slim.conv2d(
                conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
            conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
            conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
            conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
            conv3_1 = slim.conv2d(
                conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
            conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
            conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
            conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

            conv4_1 = slim.conv2d(
                conv3_4, 256, [5, 5], stride=2, scope='enc4_1')
            conv4_2 = ResnetBlock(conv4_1, 256, 5, scope='enc4_2')
            conv4_3 = ResnetBlock(conv4_2, 256, 5, scope='enc4_3')
            conv4_4 = ResnetBlock(conv4_3, 256, 5, scope='enc4_4')
            temp_1 = slim.conv2d(conv4_4, 64, [1, 1])
            refine1_1 = slim.conv2d(
                temp_1, self.chns, [5, 5], stride=1, activation_fn=None, reuse=None, scope='pred_')
            upsample_shape = tf.shape(refine1_1)
            refine1_2 = tf.image.resize_nearest_neighbor(
                refine1_1, [upsample_shape[1]*2, upsample_shape[2]*2])
            refine1_2 = slim.conv2d(
                refine1_2, 128, [1, 1], scope='refine1_2')
            refine.append(refine1_1)

            deconv3_4 = slim.conv2d_transpose(
                conv4_4, 128, [4, 4], stride=2, scope='dec3_4')
            cat3 = deconv3_4 + conv3_4

            # decoder
            deconv3_3 = ResnetBlock(cat3, 128, 5,   scope='dec3_3')
            deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
            deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')

            refine2_0 = tf.concat(
                [deconv3_1, refine1_2], axis=3, name='refine2_0')
            temp_2 = slim.conv2d(refine2_0, 64, [1, 1])
            refine2_1 = slim.conv2d(
                temp_2, self.chns, [5, 5], stride=1, activation_fn=None, reuse=True, scope='pred_')
            upsample_shape = tf.shape(refine2_1)
            refine2_2 = tf.image.resize_nearest_neighbor(
                refine1_1, [upsample_shape[1]*2, upsample_shape[2]*2])
            refine2_2 = slim.conv2d(
                refine2_2, 64, [1, 1], scope='refine2_2')
            refine.append(refine2_1)

            deconv2_4 = slim.conv2d_transpose(
                refine2_0, 64, [4, 4], stride=2, scope='dec2_4')
            cat2 = deconv2_4 + conv2_4
            deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
            deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
            deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')

            refine3_0 = tf.concat(
                [deconv2_1, refine2_2], axis=3, name='refine3_0')
            temp_3 = slim.conv2d(refine3_0, 64, [1, 1])
            refine3_1 = slim.conv2d(
                temp_3, self.chns, [5, 5], stride=1, activation_fn=None, reuse=True, scope='pred_')
            upsample_shape = tf.shape(refine3_1)
            refine3_2 = tf.image.resize_nearest_neighbor(
                refine3_1, [upsample_shape[1]*2, upsample_shape[2]*2])
            refine3_2 = slim.conv2d(
                refine3_2, 32, [1, 1], scope='refine3_2')
            refine.append(refine3_1)

            deconv1_4 = slim.conv2d_transpose(
                refine3_0, 32, [4, 4], stride=2, scope='dec1_4')
            cat1 = deconv1_4 + conv1_4
            deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
            deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
            deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')

            refine4_0 = tf.concat(
                [deconv1_1, refine3_2], axis=3, name='refine4_0')

            inp_pred = slim.conv2d(refine4_0, self.chns, [
                5, 5], activation_fn=None, reuse=True, scope='pred_')

            # edge prediction
            upsample_shape = tf.shape(deconv3_1)
            upsample_deconv3 = tf.image.resize_nearest_neighbor(
                deconv3_1, [upsample_shape[1]*4, upsample_shape[2]*4])
            upsample_deconv2 = tf.image.resize_nearest_neighbor(
                deconv2_1, [upsample_shape[1]*4, upsample_shape[2]*4])
            upsample_deconv1 = deconv1_1

            edge_feature = tf.concat(
                [upsample_deconv3, upsample_deconv2, upsample_deconv1], axis=3, name='edge_feature')

            ed = slim.conv2d(edge_feature, 1, [
                1, 1], activation_fn=None, reuse=False, scope='edge')

        return inp_pred, refine, ed

    def build_model(self):
        img_in, img_gt, img_ed = self.input_producer(self.batch_size)

        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        tf.summary.image('img_ed', im2uint8(img_ed))
        print('img_in, img_gt', 'img_ed', img_in.get_shape(),
              img_gt.get_shape(), img_ed.get_shape())

        # generator
        pred, refine, ed, all_vars = self.generator(
            img_in, reuse=False, scope=self.model_flag)
        # calculate multi-scale loss
        self.loss_total = 0
        _, hi, wi, _ = pred.get_shape().as_list()
        gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
        per_loss = self.build_perceptual_loss(pred, gt_i) * 0.01
        l1_loss = tf.reduce_mean((gt_i - pred) ** 2) * 10
        self.l1_loss = l1_loss
        self.per_loss = per_loss
        self.loss_total += per_loss
        self.loss_total += l1_loss

        #_, hi, wi, _ = ed.get_shape().as_list
        ed_i = tf.image.resize_images(img_ed, [hi, wi], method=0)
        ed_loss = tf.reduce_mean((ed_i - ed) ** 2) * 2
        self.loss_total += ed_loss

        for i in xrange(3):
            _, hi, wi, _ = refine[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - refine[i]) ** 2)*0.5
            loss = loss*(3-i)
            self.loss_total += loss
            tf.summary.image('refine_' + str(i), im2uint8(refine[i]))
            tf.summary.scalar('refine_loss_'+str(i), loss)

        tf.summary.image('out_', im2uint8(pred))
        tf.summary.image('edge_out_', im2uint8(ed))
        tf.summary.scalar('loss_per', per_loss)
        tf.summary.scalar('loss_l1', l1_loss)
        tf.summary.scalar('loss_edge', ed_loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)
        tf.summary.scalar('epoch_loss', self.epoch_loss)

        # training vars
        # all_vars = tf.trainable_variables()
        # init_var = [v for v in all_vars if 'vgg_16' not in v.name]
        self.all_vars = all_vars
        for var in all_vars:
            print(var.name)

    def build_perceptual_loss(self, gen_img, imgs):
        def styleloss(f1, f2, f3, f4):
            gen_f, _, style_f = tf.split(f1, 3, 0)
            size = tf.size(gen_f)
            style_loss = tf.nn.l2_loss(
                gram(gen_f) - gram(style_f))*2 / tf.to_float(size)

            gen_f, _, style_f = tf.split(f2, 3, 0)
            size = tf.size(gen_f)
            style_loss += tf.nn.l2_loss(gram(gen_f) -
                                        gram(style_f)) * 2 / tf.to_float(size)

            gen_f, _, style_f = tf.split(f3, 3, 0)
            size = tf.size(gen_f)
            style_loss += tf.nn.l2_loss(gram(gen_f) -
                                        gram(style_f)) * 2 / tf.to_float(size)

            gen_f, _, style_f = tf.split(f4, 3, 0)
            size = tf.size(gen_f)
            style_loss += tf.nn.l2_loss(gram(gen_f) -
                                        gram(style_f)) * 2 / tf.to_float(size)

            return style_loss
        with slim.arg_scope(vgg.vgg_arg_scope()):
            f1, f2, f3, f4, exclude = vgg.vgg_16(
                tf.concat([gen_img, imgs], axis=0))
            gen_f, img_f = tf.split(f3, 2, 0)
            content_loss = tf.nn.l2_loss(
                gen_f - img_f) / tf.to_float(tf.size(gen_f))

            return content_loss

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(
                    loss, var_list=var_list)
                unchanged_gvs = [
                    (grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad,
                            var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad,
                           var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(
                    grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(
            initial_value=self.restore_step, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=1e-6,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        # load vgg model
        vgg_model_path = '/home/opt603/Downloads/vgg_16.ckpt'
        exclude = ['vgg_16/fc6', 'vgg_16/pool4', 'vgg_16/conv5', 'vgg_16/pool5',
                   'vgg_16/fc7', 'vgg_16/global_pool', 'vgg_16/fc8/squeezed', 'vgg_16/fc8']
        vgg_vars = slim.get_variables_to_restore(
            include=['vgg_16'], exclude=exclude)
        # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
        init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
        init_fn(self.sess)
        # tf.initialize_variables(var_list=vgg_init_var)
        print('vgg s weights load done')
        self.saver = tf.train.Saver(
            max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            self.train_dir, sess.graph, flush_secs=30)

        self.load(sess, self.restore_dir, step=self.restore_step)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(
                loss_total_val), 'Model diverged with loss = NaN'

            self.epoch_loss += loss_total_val

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = (%.5f; %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step,
                                    loss_total_val, self.epoch_loss, examples_per_sec, sec_per_batch))

            if step % self.data_list == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
                self.epoch_loss = 0

            # Save the model checkpoint periodically.
            if step > self.max_steps/2:
                if step % 1000 == 0 or step == self.max_steps or loss_total_val < 0.016:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'checkpoints')
                    self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(
            checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(
                checkpoint_dir, 'checkpoints', ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_datalist = open('test_datalist.txt', 'rt').read().splitlines()
        test_datalist = list(map(lambda x: x.split(' '), test_datalist))

        imgsName = [x[0] for x in test_datalist]

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(
            shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs, _, _, _ = self.generator(
            inputs, reuse=False, scope=self.model_flag)

        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=self.load_step)

        for imgName in imgsName:
            blur = scipy.misc.imread(imgName)
            split_name = imgName.split('/')
            path_temp = os.path.join(
                output_path, self.model_flag + '_' + self.exp_num, split_name[-3], 'sharp')
            if not os.path.exists(path_temp):
                os.makedirs(path_temp)
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(
                    blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(
                    blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model is not 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (imgName, duration))
            res = deblur
            if self.args.model is not 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(path_temp, split_name[-1]), res)
