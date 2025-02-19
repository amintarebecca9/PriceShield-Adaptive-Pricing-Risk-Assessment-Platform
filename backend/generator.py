import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_dim=latent_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='linear')  # Output: simulated market features
    ])
    return model

def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=input_shape),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_gan(generator, discriminator, latent_dim, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        
        # Create real market data samples (for demo, using random data)
        real_data = np.random.uniform(-1, 1, (batch_size, 8))
        
        # Labels for real and fake data
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_data, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_y)
        
        # Train generator via combined model
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        # Freeze discriminator's weights during generator training
        discriminator.trainable = False
        combined_input = noise
        combined_loss = combined_model.train_on_batch(combined_input, valid_y)
        discriminator.trainable = True
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss_real={d_loss_real}, D_loss_fake={d_loss_fake}, G_loss={combined_loss}")

# Prepare combined model for generator training
latent_dim = 10
generator = build_generator(latent_dim)
discriminator = build_discriminator((8,))
# Build combined model
discriminator.trainable = False
combined_input = tf.keras.Input(shape=(latent_dim,))
generated_data = generator(combined_input)
validity = discriminator(generated_data)
combined_model = tf.keras.Model(combined_input, validity)
combined_model.compile(optimizer='adam', loss='binary_crossentropy')

