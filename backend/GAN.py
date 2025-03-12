import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------
#  Model Definitions
# ---------------------

def build_generator(latent_dim):
    # Generator built using the Functional API
    inputs = Input(shape=(latent_dim,))
    x = Dense(16)(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(16)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    # Output layer: a single market factor value (linear activation)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs, name="generator")
    return model

def build_discriminator():
    # Discriminator built using the Functional API
    inputs = Input(shape=(1,))
    x = Dense(16)(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dense(8)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name="discriminator")
    return model

def build_gan(generator, discriminator):
    # Freeze discriminator weights for GAN training
    discriminator.trainable = False
    latent_dim = generator.input_shape[1]
    gan_input = Input(shape=(latent_dim,))
    generated_output = generator(gan_input)
    gan_output = discriminator(generated_output)
    gan = Model(gan_input, gan_output, name="gan")
    return gan

# ---------------------
#  Custom Training Loop
# ---------------------

def train_gan_custom(generator, discriminator, gan, latent_dim, epochs=5000, batch_size=32, sample_interval=500):
    # Create dummy real market factor data (replace with real data if available)
    real_market_factors = np.random.normal(0, 1, (1000, 1)).astype(np.float32)
    
    # Optimizers for discriminator and generator
    optimizer_d = Adam(learning_rate=0.0002)
    optimizer_g = Adam(learning_rate=0.0002)
    
    # Loss function: Binary Crossentropy
    bce = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator on Real Samples
        # ---------------------
        
        # Select a random batch of real samples
        idx = np.random.randint(0, real_market_factors.shape[0], batch_size)
        real_samples = real_market_factors[idx]
        
        # Label smoothing for real samples
        real_labels = 0.9 * np.ones((batch_size, 1), dtype=np.float32)
        
        with tf.GradientTape() as tape:
            pred_real = discriminator(real_samples, training=True)
            loss_real = bce(real_labels, pred_real)
        grads_real = tape.gradient(loss_real, discriminator.trainable_variables)
        # Filter out any None gradients
        grads_vars_real = [(g, v) for g, v in zip(grads_real, discriminator.trainable_variables) if g is not None]
        if grads_vars_real:
            optimizer_d.apply_gradients(grads_vars_real)
        
        # ---------------------
        #  Train Discriminator on Fake Samples
        # ---------------------
        
        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        fake_samples = generator(noise, training=True)
        fake_labels = np.zeros((batch_size, 1), dtype=np.float32)
        
        with tf.GradientTape() as tape:
            pred_fake = discriminator(fake_samples, training=True)
            loss_fake = bce(fake_labels, pred_fake)
        grads_fake = tape.gradient(loss_fake, discriminator.trainable_variables)
        grads_vars_fake = [(g, v) for g, v in zip(grads_fake, discriminator.trainable_variables) if g is not None]
        if grads_vars_fake:
            optimizer_d.apply_gradients(grads_vars_fake)
        
        d_loss = 0.5 * (loss_real + loss_fake)
        
        # ---------------------
        #  Train Generator via GAN
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        valid_labels = np.ones((batch_size, 1), dtype=np.float32)  # generator wants discriminator to output 1
        
        with tf.GradientTape() as tape:
            pred = gan(noise, training=True)
            g_loss = bce(valid_labels, pred)
        grads_g = tape.gradient(g_loss, generator.trainable_variables)
        grads_vars_g = [(g, v) for g, v in zip(grads_g, generator.trainable_variables) if g is not None]
        if grads_vars_g:
            optimizer_g.apply_gradients(grads_vars_g)
        
        # Optionally print the progress at intervals
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss.numpy():.4f}] [G loss: {g_loss.numpy():.4f}]")
    
    return generator

# ---------------------
#  Main Execution
# ---------------------

if __name__ == '__main__':
    latent_dim = 10
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # It's important to compile the discriminator before training it separately
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0002),
        metrics=['accuracy']
    )
    
    # Note: We don't compile the GAN since we'll use a custom training loop
    # Train using the custom loop
    generator = train_gan_custom(generator, discriminator, gan, latent_dim, epochs=5000, batch_size=32, sample_interval=500)
    
    # Save the trained generator model for later use in your Flask app
    generator.save("GAN/gan.h5")
    print("Generator saved as 'gan.h5'")
