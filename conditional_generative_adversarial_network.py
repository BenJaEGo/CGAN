from generator import *
from discriminator import *


class ConditionalGenerativeAdversarialNetwork(object):
    def __init__(self, n_input, n_generator_units, n_discriminator_units, n_latent, n_label, dis_lr, gen_lr, lam):
        self._n_input = n_input
        self._n_generator_units = n_generator_units
        self._n_discriminator_units = n_discriminator_units
        self._n_latent = n_latent
        self._n_label = n_label

        self._dis_lr = tf.Variable(dis_lr)
        self._gen_lr = tf.Variable(gen_lr)

        self._x_pl = tf.placeholder(tf.float32, shape=[None, n_input], name='x_pl')
        self._z_pl = tf.placeholder(tf.float32, shape=[None, n_latent], name='z_pl')
        self._y_pl = tf.placeholder(tf.float32, shape=[None, n_label], name='y_pl')

        self._generator = Generator(n_latent, n_generator_units, n_input, n_label, tf.nn.relu, tf.nn.sigmoid)
        self._discriminator = Discriminator(n_input, n_discriminator_units, 1, n_label, tf.nn.relu, tf.nn.sigmoid)

        self._generated_sample = self._generator.forward(self._z_pl, self._y_pl)
        fake_probability = self._discriminator.forward(self._generated_sample, self._y_pl)
        real_probability = self._discriminator.forward(self._x_pl, self._y_pl)

        discriminator_loss = -tf.reduce_mean(tf.log(real_probability) + tf.log(1.0 - fake_probability))
        generator_loss = -tf.reduce_mean(tf.log(fake_probability))

        self._discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self._dis_lr, name="ADAM_optimizer_dis")
        self._generator_optimizer = tf.train.AdamOptimizer(learning_rate=self._gen_lr, name="ADAM_optimizer_gen")

        discriminator_wd_loss = lam * self._discriminator.wd_loss
        generator_wd_loss = lam * self._generator.wd_loss

        self._discriminator_loss = discriminator_loss + discriminator_wd_loss
        self._generator_loss = generator_wd_loss + generator_loss

        dis_global_step = tf.Variable(0, name="dis_global_step", trainable=False)
        gen_global_step = tf.Variable(0, name="gen_global_step", trainable=False)

        dis_trainable_variables = self._discriminator.parameters
        gen_trainable_variables = self._generator.parameters

        dis_grads = tf.gradients(ys=self._discriminator_loss, xs=dis_trainable_variables)
        gen_grads = tf.gradients(ys=self._generator_loss, xs=gen_trainable_variables)

        self._dis_train_op = self._discriminator_optimizer.apply_gradients(
            grads_and_vars=zip(dis_grads, dis_trainable_variables),
            global_step=dis_global_step,
            name="dis_train_op")
        self._gen_train_op = self._generator_optimizer.apply_gradients(
            grads_and_vars=zip(gen_grads, gen_trainable_variables),
            global_step=gen_global_step,
            name="gen_train_op")

    @property
    def x_pl(self):
        return self._x_pl

    @property
    def z_pl(self):
        return self._z_pl

    @property
    def y_pl(self):
        return self._y_pl

    @property
    def dis_loss(self):
        return self._discriminator_loss

    @property
    def dis_train_op(self):
        return self._dis_train_op

    @property
    def gen_loss(self):
        return self._generator_loss

    @property
    def gen_train_op(self):
        return self._gen_train_op

    @property
    def generated_sample(self):
        return self._generated_sample
