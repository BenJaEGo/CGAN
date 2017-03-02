from tensorflow.examples.tutorials.mnist import input_data
from conditional_generative_adversarial_network import *
from tf_tools import *
from vis_tools import *
import os


def run_training():
    save_path = 'out'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_input = 784
    n_label = 10
    n_generator_units = [200]
    n_discriminator_units = [200]
    n_latent = 100
    lam = 0.0001
    lr = 0.001

    desired_label = 5

    max_epoch = 4000
    batch_size = 100
    n_sample, n_dims = mnist.train.images.shape
    n_batch_each_epoch = n_sample // batch_size

    graph = tf.Graph()

    with graph.as_default():

        model = ConditionalGenerativeAdversarialNetwork(n_input, n_generator_units, n_discriminator_units, n_latent,
                                                        n_label, lr, lr, lam)

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(max_epoch):
                aver_dis_loss = 0.0
                aver_gen_loss = 0.0
                for step in range(n_batch_each_epoch):

                    x, y = mnist.train.next_batch(batch_size)
                    tr_dis_loss, _ = sess.run(
                        fetches=[model.dis_loss, model.dis_train_op],
                        feed_dict={model.x_pl: x,
                                   model.y_pl: y,
                                   model.z_pl: sample_latent_variables_uniform(batch_size, n_latent)}
                    )

                    tr_gen_loss, _ = sess.run(
                        fetches=[model.gen_loss, model.gen_train_op],
                        feed_dict={
                            model.y_pl: y,
                            model.z_pl: sample_latent_variables_uniform(batch_size, n_latent)}
                    )

                    aver_dis_loss += tr_dis_loss
                    aver_gen_loss += tr_gen_loss

                print("epoch %d, tr_dis_loss %f, tr_gen_loss %f" %
                      (epoch, aver_dis_loss / n_batch_each_epoch, aver_gen_loss / n_batch_each_epoch))

                y = np.zeros(shape=[16, n_label])
                y[:, desired_label] = 1

                samples = sess.run(fetches=[model.generated_sample],
                                   feed_dict={
                                       model.y_pl: y,
                                       model.z_pl: sample_latent_variables_uniform(16, n_latent)})
                # print(samples[0].shape)
                fig = visualize_generate_samples(samples[0])
                plt.savefig('{path}/epoch_{epoch}.png'.format(
                    path=save_path, epoch=epoch), bbox_inches='tight')
                plt.close(fig)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
