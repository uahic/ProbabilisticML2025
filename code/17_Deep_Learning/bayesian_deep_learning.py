import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib

matplotlib.use("TkAgg")  # Tk
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# import pandas as pd
import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve
from sklearn.datasets import make_moons
from tqdm import tqdm, trange
from matplotlib.colors import LinearSegmentedColormap
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from probML2025.shared import rgb, cmap_wr, cmap_wd, cmap_bwo, KEY
from probML2025.gaussians import Gaussian
from probML2025.gp import GaussianProcess
from collections.abc import Callable
from functools import partial

jax.config.update("jax_enable_x64", True)


bwo = LinearSegmentedColormap.from_list(
    "bwo", colors=[rgb.tue_blue, (1, 1, 1), rgb.tue_orange], N=1024
)
gw = LinearSegmentedColormap.from_list(
    "gw", colors=[(1, 1, 1), "w", rgb.tue_dark], N=1024
)

####### Prior
LAYER_SIZES = [2, 128, 64, 1]
PARAM_SCALE = 0.1

LEARNING_RATE = 1e-3
NUM_EPOCHS = 300
LOG_FREQUENCY = 100
BATCH_SIZE = 64

KEY = random.key(3)

import equinox as eqx


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, key: PRNGKeyArray, layer_sizes: list[int]) -> None:
        keys = random.split(key, len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def __call__(self, x: Float[Array, "B D"]) -> Float[Array, "B 1"]:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x


KEY, model_key = random.split(KEY)
model = MLP(model_key, layer_sizes=LAYER_SIZES)


from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=10 * BATCH_SIZE, noise=0.1, random_state=0)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 80:20 split
Xtrain, Xtest = X[: 8 * BATCH_SIZE], X[8 * BATCH_SIZE :]
Ytrain, Ytest = Y[: 8 * BATCH_SIZE], Y[8 * BATCH_SIZE :]


def plot_decision_boundary(
    model: MLP,
    Xtrain: Float[Array, "B 2"],
    Ytrain: Int[Array, " B"],
    Xtest: Float[Array, "B 2"],
    Ytest: Int[Array, " B"],
) -> None:
    x1 = jnp.linspace(-5, 5, 200)
    x2 = jnp.linspace(-5, 5, 200)
    X1, X2 = jnp.meshgrid(x1, x2)
    X_grid = jnp.stack([X1.flatten(), X2.flatten()], axis=1)

    # make prediction for each point on the X_grid
    Y_grid = jax.nn.sigmoid(jax.vmap(model)(X_grid)).flatten()

    # Create figure and plot decision boundary
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot decision boundary as contour
    Y_grid_reshaped = Y_grid.reshape(
        X1.shape
    )  # Reshape Y_grid to unfolded grid size (200,200)
    contour = ax.contourf(X1, X2, Y_grid_reshaped, levels=100, cmap=gw, alpha=0.8)
    ax.contour(X1, X2, Y_grid_reshaped, levels=[0.5], linewidths=2)
    plt.colorbar(contour, ax=ax, label="Predicted Probability")

    # Plot training data
    ax.scatter(
        Xtrain[Ytrain == 0, 0],
        Xtrain[Ytrain == 0, 1],
        c="blue",
        marker="o",
        s=50,
        edgecolor="None",
        vmin=0,
        facecolor="None",
        vmax=1,
        label="Train Class 0",
        alpha=0.7,
    )
    ax.scatter(
        Xtrain[Ytrain == 1, 0],
        Xtrain[Ytrain == 1, 1],
        c="orange",
        marker="s",
        s=50,
        edgecolor="None",
        vmin=0,
        facecolor="None",
        vmax=1,
        label="Train Class 1",
        alpha=0.7,
    )

    # Plot test data
    ax.scatter(
        Xtest[Ytest == 0, 0],
        Xtest[Ytest == 0, 1],
        c="blue",
        marker="^",
        s=100,
        edgecolor="black",
        facecolor="None",
        vmin=0,
        vmax=1,
        label="Test Class 0",
        alpha=0.9,
    )
    ax.scatter(
        Xtest[Ytest == 1, 0],
        Xtest[Ytest == 1, 1],
        c="orange",
        marker="v",
        s=100,
        edgecolor="black",
        facecolor="None",
        vmin=0,
        vmax=1,
        label="Test Class 1",
        alpha=0.9,
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary with Training and Test Data")

    plt.legend(loc="lower right")
    plt.show()


plot_decision_boundary(model, Xtrain, Ytrain, Xtest, Ytest)


## Training
import torch.utils.data as data


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, tuple)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.D = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]


train_dataset = MyDataset(Xtrain, Ytrain)
train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=numpy_collate
)

test_dataset = MyDataset(Xtest, Ytest)
test_loader = data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=numpy_collate
)

data_input, data_output = next(iter(train_loader))


#### Optimizer

import optax


@eqx.filter_jit
def loss_fn(
    model: MLP, x: Float[Array, "B 2"], y: Int[Array, " B"]
) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)[:, 0]
    return optax.losses.sigmoid_binary_cross_entropy(preds, y).mean()


@eqx.filter_jit
def compute_accuracy(
    model: MLP, x: Float[Array, "B 2"], y: Int[Array, " B"]
) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)[:, 0]
    argmax_preds = jnp.astype(preds > 0, jnp.int64)
    return jnp.mean(y == argmax_preds)


def test(
    model: MLP, test_loader: data.DataLoader
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    avg_loss = jnp.zeros(())
    avg_accuracy = jnp.zeros(())

    for batch_x, batch_y in test_loader:
        avg_loss += loss_fn(model, batch_x, batch_y)
        avg_accuracy += compute_accuracy(model, batch_x, batch_y)

    return avg_loss / len(test_loader), avg_accuracy / len(test_loader)


batch_loss, batch_grads = eqx.filter_value_and_grad(loss_fn)(
    model, data_input, data_output
)

batch_accuracy = compute_accuracy(model, data_input, data_output)
print(f"{batch_accuracy = }, {batch_loss = }")

test_loss, test_accuracy = test(model, test_loader)
print(f"{test_loss = }, {test_accuracy = }")


optimizer = optax.adam(learning_rate=LEARNING_RATE)


def train(
    model: MLP,
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    optimizer: optax.GradientTransformation,
    num_epochs: int,
    log_frequency: int,
) -> MLP:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: MLP,
        opt_state: PyTree,
        batch_x: Float[Array, "B 2"],
        batch_y: Int[Array, " B"],
    ) -> tuple[MLP, PyTree, Float[Array, ""]]:
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch_x, batch_y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    loss_log = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        for batch_x, batch_y in train_loader:
            model, opt_state, train_loss = make_step(model, opt_state, batch_x, batch_y)
            loss_log.append(train_loss)

        test_loss, test_accuracy = test(model, test_loader)

        if epoch % log_frequency == 0:
            pbar.set_postfix(
                train_loss=train_loss.item(),
                test_loss=test_loss.item(),
                test_accuracy=test_accuracy.item(),
            )

    return model, loss_log


model, loss_log = train(
    model, train_loader, test_loader, optimizer, NUM_EPOCHS, LOG_FREQUENCY
)

test_loss, test_accuracy = test(model, test_loader)
print(f"Final test_loss = {test_loss:.2e}, test_accuracy = {test_accuracy:.2e}")

plot_decision_boundary(model, Xtrain, Ytrain, Xtest, Ytest)


fig, ax = plt.subplots()
ax.plot(loss_log, color=rgb.tue_blue)

# Introducing Bayesian uncertainty via Laplax
from laplax.curv.cov import create_posterior_fn
from laplax.curv.ggn import create_ggn_mv # curvature operator (generalized gaus-newton matrix, multi-variate output)
from laplax.eval.pushforward import set_posterior_gp_kernel

# Split model into parameters and static fiels (e.g. layers)
params_final, static = eqx.partition(model, eqx.is_array)


def model_fn(params, input):
    new_model = eqx.combine(static, params)
    return new_model(input)

# we will build a Laplace approximation from the entire training set. If this is not feasible,
# one can also subsample, but then we get an upper bound on the epistemic uncertainty
# (lower bound on confidence) of the model.
batch = {"input": Xtrain, "target": Ytrain}

# create the GGN matrix-vector product function
ggn_mv = create_ggn_mv(
    model_fn,
    params_final,
    loss_fn="binary_cross_entropy",
    # vmap_over_data=True,
    data=batch,
)

# this object returns the Laplace posterior
posterior_fn = create_posterior_fn(
    "full",
    mv=ggn_mv,
    layout=params_final,
)

# we define prior precision (this need not be the same as the training regularizer, because
# one may need to do weird things with the "prior" to make the optimizer converge well.
# Also, let's be honest, regularizers, while mathematically equivalent to a prior,
# are usually not designed with interpretation in mind).
prior_arguments = {"prior_prec": 1.0}

# now laplax can return the posterior kernel of our linearized model:
laplax_gp_kernel, _ = set_posterior_gp_kernel(
    model_fn=model_fn,
    mean=params_final,
    posterior_fn=posterior_fn,
    prior_arguments=prior_arguments,
    dense=True,
    output_layout=1,
)

laplax_gp_kernel = jax.jit(laplax_gp_kernel)

# the mean function is just the model, evaluated at the MAP parameters
@eqx.filter_jit
def mean_func(x: Float[Array, "B 2"]) -> Float[Array, "B"]:
    return jax.vmap(model)(x)[:, 0]


def vectorized_laplace_kernel(
    a: Float[Array, "B 2"],
    b: Float[Array, "B 2"],
):
    vectorized_kernel = jnp.vectorize(
        laplax_gp_kernel,
        signature="(d),(d)->(c,c)",
    )(
        a, b
    )

    return vectorized_kernel[..., 0, 0]  # discard scalar dimensions


# and here it is, our GP posterior:
laplax_posterior = GaussianProcess(mean_func, vectorized_laplace_kernel)

# and we make a plot
key = random.key(0)

x1 = jnp.linspace(-5, 5, 50)
x2 = jnp.linspace(-5, 5, 50)
X1, X2 = jnp.meshgrid(x1, x2)
X_grid = jnp.stack([X1.flatten(), X2.flatten()], axis=1)

mX_laplax = laplax_posterior.m(X_grid).reshape(X1.shape)
vX_laplax = laplax_posterior.k(X_grid, X_grid).reshape(X1.shape)

prediction_laplax = jax.nn.sigmoid(
    mX_laplax / jnp.sqrt(1 + jnp.pi / 8 * vX_laplax)
)

sample = laplax_posterior(X_grid).sample(key=key, num_samples=4)

# Use standard matplotlib style for presentations
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 3)})
fig, axs = plt.subplots(1, 4)

cp = axs[0].contourf(
    X1, X2, jax.nn.sigmoid(mX_laplax),
    cmap=bwo, alpha=0.5, levels=10
)
fig.colorbar(ax=axs[0], mappable=cp)
axs[0].set_title("point estimate")

cp = axs[1].contourf(
    X1, X2, mX_laplax,
    cmap=bwo, alpha=0.5, levels=10
)
fig.colorbar(ax=axs[1], mappable=cp)
axs[1].set_title("logit")

cp = axs[2].contourf(
    X1, X2, jnp.sqrt(vX_laplax),
    cmap=gw, alpha=0.5, levels=100, vmin=0, vmax=50
)
axs[2].set_title("stddev.")
fig.colorbar(ax=axs[2], mappable=cp)

cp = axs[3].contourf(
    X1, X2, prediction_laplax,
    cmap=bwo, alpha=0.5, levels=10, vmin=0, vmax=1
)
fig.colorbar(ax=axs[3], mappable=cp)
axs[3].set_title("prediction")
for ax in axs:
    ax.scatter(
        Xtrain[:, 0],
        Xtrain[:, 1],
        c=Ytrain,
        cmap=bwo,
        s=3,
        alpha=0.9,
        edgecolor="None",
        vmin=0,
        vmax=1,
        label="train",
    )

fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
for i, ax in enumerate(axs):
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.contour(
        X1,
        X2,
        jax.nn.sigmoid(sample[i, :].reshape(X1.shape)),
        cmap=bwo,
        alpha=0.3,
        levels=100,
        vmin=0,
        vmax=1,
    )

    for ax in axs:
        sc = ax.scatter(
            Xtrain[:, 0],
            Xtrain[:, 1],
            c=Ytrain,
            cmap=bwo,
            s=3,
            alpha=0.9,
            edgecolor="None",
            vmin=0,
            vmax=1,
            label="train",
        )

fig.colorbar(sc, ax=axs[3])
fig.show()