'''
IMPORTANT: DO NOT modify the existing stencil code or else it may not be compatible with the Autograder
If you find any problems, report to TA's.
'''
import tensorflow as tf
layers = tf.layers
import numpy as np
from scipy.misc import imsave
import os
import argparse
import tensorflow.contrib.gan as gan
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='DCGAN')

parser.add_argument('--img-dir', type=str, default='/course/cs1470/asgn/dcgan/celebA',
  help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='./output',
  help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
  help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
  help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=100,
  help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=128,
  help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
  help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=10,
  help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
  help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
  help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
  help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=10,
  help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=100,
  help='Save the state of the network after every [this many] training iterations')

args = parser.parse_args()

## --------------------------------------------------------------------------------------

# Numerically stable logarithm function
def log(x):
    return tf.log(tf.maximum(x, 1e-5))

## --------------------------------------------------------------------------------------

class Model:

    def __init__(self, image_batch, g_input_z):
        self.image_batch = image_batch
        self.g_input_z = g_input_z

        ### YOUR CODE GOES HERE
        # Finish setting up the TF graph:
        #  - Build the generator graph
        #  - Build the discriminator graph with 2 inputs, one for real image input, one for fake
        #  - You might want to create helper function(s) to help accomplish the task

        self.logits_real = None
        self.logits_fake = None

        self.g_output = self.generator(self.g_input_z)

        with tf.variable_scope("discriminator") as scope:
            self.logits_real = self.discriminator(self.image_batch)

        with tf.variable_scope("discriminator", reuse=True) as scope:
            self.logits_fake = self.discriminator(self.g_output)

        # Declare losses, optimizers(trainers) and fid for evaluation
        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        self.fid = self.fid_function()

    def generator(self, z):
        with tf.variable_scope("generator"):
            W = tf.Variable(tf.random_normal([args.batch_size, args.z_dim, 16*512]))
            init = tf.reshape(tf.matmul(z, W), [args.batch_size,4,4,512])
            init = tf.layers.batch_normalization(init)
            init = tf.nn.relu(init)

            deconv1 = layers.conv2d_transpose(init, 256, [5,5], (2,2), padding='same')
            deconv1 = tf.layers.batch_normalization(deconv1)
            deconv1 = tf.nn.relu(deconv1)

            deconv2 = layers.conv2d_transpose(deconv1, 128, [5,5], (2,2), padding='same')
            deconv2 = tf.layers.batch_normalization(deconv2)
            deconv2 = tf.nn.relu(deconv2)

            deconv3 = layers.conv2d_transpose(deconv2, 64, [5,5], (2,2), padding='same')
            deconv3 = tf.layers.batch_normalization(deconv3)
            deconv3 = tf.nn.relu(deconv3)

            deconv4 = layers.conv2d_transpose(deconv3, 3, [5,5], (2,2), padding='same')
            #deconv4 = tf.layers.batch_normalization(deconv4)
            return tf.nn.tanh(deconv4)

    def discriminator(self, x):
        conv1 = layers.conv2d(x, 64, [5,5], (2,2), padding='same')
        #conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = layers.conv2d(conv1, 128, [5,5], (2,2), padding='same')
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.leaky_relu(conv2)

        conv3 = layers.conv2d(conv2, 256, [5,5], (2,2), padding='same')
        conv3 = tf.layers.batch_normalization(conv3)
        conv3 = tf.nn.leaky_relu(conv3)

        conv4 = layers.conv2d(conv3, 512, [5,5], (2,2), padding='same')
        conv4 = tf.layers.batch_normalization(conv4)
        conv4 = tf.nn.leaky_relu(conv4)

        return layers.dense( tf.reshape(conv4, [args.batch_size, 4*4*512]), 1, tf.sigmoid)

    # Training loss for Generator
    def g_loss_function(self):
        #g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.logits_fake), logits=self.logits_fake))
        g_loss = tf.reduce_mean(-log(self.logits_fake))
        return g_loss

    # Training loss for Discriminator
    def d_loss_function(self):
        #d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.logits_real), logits=self.logits_real)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.logits_fake), logits=self.logits_fake))
        d_loss = 0.5*tf.reduce_mean(-log(1-self.logits_fake) - log(self.logits_real))
        return d_loss

    # Optimizer/Trainer for Generator
    def g_trainer(self):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        g_train = tf.train.AdamOptimizer(args.learn_rate, args.beta1).minimize(self.g_loss, var_list=g_vars)
        return g_train

    # Optimizer/Trainer for Discriminator
    def d_trainer(self):
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        d_train = tf.train.AdamOptimizer(args.learn_rate, args.beta1).minimize(self.d_loss, var_list=d_vars)
        return d_train

    # For evaluating the quality of generated images
    # Frechet Inception Distance measures how similar the generated images are to the real ones
    # https://nealjean.com/ml/frechet-inception-distance/
    # Lower is better
    def fid_function(self):
        INCEPTION_IMAGE_SIZE = (299, 299)
        real_resized = tf.image.resize_images(self.image_batch, INCEPTION_IMAGE_SIZE)
        fake_resized = tf.image.resize_images(self.g_output, INCEPTION_IMAGE_SIZE)
        return gan.eval.frechet_classifier_distance(real_resized, fake_resized, gan.eval.run_inception)

# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dirname, batch_size=128,
    shuffle_buffer_size=250000, n_threads=2):

    # Function used to load and pre-process image files
    # (Have to define this ahead of time b/c Python does allow multi-line
    #    lambdas, *grumble*)
    def load_and_process_image(filename):
        # Load image
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List filenames
    dir_path= dirname + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)
    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image,
        num_parallel_calls=n_threads)
    # Create batch, dropping the final one which has less than
    #    batch_size elements
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))
    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset that we can
    #    re-initialize for each new epoch
    return dataset.make_initializable_iterator()

## --------------------------------------------------------------------------------------

# 'image_batch' is a TF node carrying a batch of images loaded from disk
dataset_iterator = load_image_batch(args.img_dir,
    batch_size=args.batch_size,
    n_threads=args.num_data_threads)
image_batch = dataset_iterator.get_next()

### YOUR CODE GOES HERE
# Finish setting up the DCGAN model
# - Set up a tf placeholder for generator input (g_input_z)
# - Initialize DCGAN model (graph) by using the Model class above

g_input_z = tf.placeholder(tf.float32, [args.batch_size, 1, args.z_dim])

model = Model(image_batch, g_input_z)

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# For saving/loading models
saver = tf.train.Saver()

## --------------------------------------------------------------------------------------

# Load the last saved checkpoint during training or used by test
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

# Train the model
def train():
    # Training loop
    for epoch in range(0, args.num_epochs):
        print('========================== EPOCH %d  ==========================' % epoch)

        # Initialize the dataset iterator for this epoch (this shuffles the
        #    data differently every time)
        sess.run(dataset_iterator.initializer)

        # Loop over our data until we run out
        iteration = 0
        try:
            while True:

                #### YOUR CODE GOES HERE
                z = np.random.uniform(-1.,1.,(args.batch_size, 1, args.z_dim,))
                _, loss_d = sess.run([model.d_train, model.d_loss], feed_dict= {g_input_z: z})

                for i in range(args.num_gen_updates):
                    _, loss_g = sess.run([model.g_train, model.g_loss], feed_dict= {g_input_z: z})

                # Print losses
                if iteration % args.log_every == 0:
                    print('Iteration %d: Gen loss = %g | Discrim loss = %g' % \
			(iteration, loss_g, loss_d))

                # Save
                if iteration % args.save_every == 0:
                    saver.save(sess, './dcgan_saved_model')
                iteration += 1
        except tf.errors.OutOfRangeError:
            # Triggered when the iterator runs out of data
            pass

        # Save at the end of the epoch, too
        saver.save(sess, './dcgan_saved_model')

        # Also, print the inception distance
        sess.run(dataset_iterator.initializer)
        #### YOUR CODE GOES HERE
        z = np.random.uniform(-1.,1.,(args.batch_size, 1, args.z_dim,))
        fid_ = sess.run(model.fid, feed_dict= {g_input_z: z})  # Use sess.run to get the inception distance value defined above
        print('**** INCEPTION DISTANCE: %g ****' % fid_)


# Test the model by generating some samples from random latent codes
def test():

    ### YOUR CODE GOES HERE
    z = np.random.uniform(-1.,1.,(args.batch_size, 1, args.z_dim,))
    gen_img_batch = sess.run(model.g_output, feed_dict={g_input_z: z})     # Replace 'None' with code to sample a batch of random images

    ### Below, we've already provided code to save these generated images to files on disk

    # Rescale the image from (-1, 1) to (0, 255)
    gen_img_batch = ((gen_img_batch / 2) - 0.5) * 255
    # Convert to uint8
    gen_img_batch = gen_img_batch.astype(np.uint8)
    # Save images to disk
    for i in range(0, args.batch_size):
        img_i = gen_img_batch[i]
        s = args.out_dir+'/'+str(i)+'.png'
        imsave(s, img_i)

## --------------------------------------------------------------------------------------

# Ensure the output directory exists
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.restore_checkpoint or args.mode == 'test':
    load_last_checkpoint()

if args.mode == 'train':
    train()
if args.mode == 'test':
    test()
