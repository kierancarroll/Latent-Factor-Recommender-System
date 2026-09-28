import numpy as np

def latent_factor_loss(ratings, U, V):
    """ Return the latent factor loss
        ratings: (num_ratings, 3) numpy ndarray with one row per rating
            Each row has three columns containing integers: [user_index, book_index, rating]
            Indices are zero-indexed.
        U: (num_users, K) numpy ndarray with float values
        V: (num_items, K) numpy ndarray with float values

        Returns a float with the resulting loss values
    """
    loss = 0
    N = ratings.shape[0]

    for i, j, r in ratings:
        error = (r - U[i] @ V[j])
        loss += 1/2*error**2

    loss = loss/N

    return loss


def latent_factor_sgd(ratings, num_users, num_items, K, alpha, num_epoch, print_loss=False):
    """ Return the user matrix, U, and item matrix, V, resulting from solving the
        latent factor problem with stochastic gradient descent.

        ratings: (num_ratings, 3) numpy ndarray with one row per rating
            Each row has three columns containing integers: [user_index, book_index, rating]
            Indices are zero-indexed.
        num_users: int representing the number of users
        num_items: int representing the number of items
        K: int representing the number of dimensions in the low dimensional embedding
        alpha: float representing the SGD learning rate
        num_epoch: int representing the number of epochs to train

        Returns tuple of length 2: (U, V)
        where:
            U: (num_users, K) numpy ndarray with float values
            V: (num_items, K) numpy ndarray with float values
    """
    np.random.seed(315)
    U = np.random.normal(0, 1, size=(num_users, K))
    V = np.random.normal(0, 1, size=(num_items, K))

    for epoch in range(num_epoch):
        for i, j, r in ratings:
            gradient_loss_u = (-(r - U[i] @ V[j]) * V[j])
            gradient_loss_v = (-(r - U[i] @ V[j]) * U[i])

            U[i] = U[i] - alpha * gradient_loss_u
            V[j] = V[j] - alpha * gradient_loss_v

        if print_loss and epoch % (num_epoch/10) == (num_epoch/10 - 1):
            loss = latent_factor_loss(ratings, U, V)
            print('Training loss epoch {} of {}: {}'.format(epoch+1, num_epoch, loss))

    return (U, V)