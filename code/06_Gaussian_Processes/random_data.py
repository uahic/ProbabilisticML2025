import jax
import jax.numpy as jnp


def sample_noisy_linear(key, n, a=2.0, b=1.0, sigma=0.5, x_range=(-5.0, 5.0)):
    key_x, key_noise = jax.random.split(key)

    X = jax.random.uniform(key_x, shape=(n,), minval=x_range[0], maxval=x_range[1])

    noise = sigma * jax.random.normal(key_noise, shape=(n,))
    Y = a * X + b + noise

    return X, Y


if __name__ == "__main__":
    # example usage
    key = jax.random.PRNGKey(0)
    X, Y = sample_noisy_linear(key, n=1000, a=1.5, b=-0.5, sigma=0.3)
